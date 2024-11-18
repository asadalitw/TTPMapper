import torch
from torch.nn import Softmax
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import requests
from collections import defaultdict

def classification(sentences, model_path, tokenizer_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    predicted_logits = outputs.logits
    softmax = Softmax(dim=1)
    predicted_probabilities = softmax(predicted_logits)
    predicted_labels = torch.argmax(predicted_probabilities, dim=1)
    
    return predicted_labels, predicted_probabilities

def query_gpt_api(sentence):
    api_endpoint = 'http://192.168.1.198:3000/azure/api/v1/chat'
    my_prompt = f"Sentence: {sentence}\n What MITRE techniques do you infer from this sentence? Give me top 3 techniques with probability. \nYour response should have the following format: MITRE techniques number, MITRE techniques probability in percentage. Here is an example response: \n\nMITRE techniques number: T1499, T1498, T1071\nMITRE techniques probability in percentage: T1499 (70%), T1498 (20%), T1071 (10%)"
    
    payload = {
        "model": "gpt-4o",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a cybersecurity expert"},
            {"role": "user", "content": my_prompt}
        ] 
    }
    
    response = requests.post(api_endpoint, json=payload)
    
    if response.status_code == 200:
        print("+"*80)
        print("\nDID GPT GIVE RESPONSE\n")
        print(response.json())
        return response.json()
    else:
        print(f"Failed to get response from GPT API. Status code: {response.status_code}")
        return None

def parse_gpt_response(gpt_response):
    techniques = []
    probabilities = []

    try:
        answer = gpt_response['answer']
        partss = answer.split('\n')
    
        techniques_part_new = partss[2].replace('MITRE techniques number: ', '').strip().split(', ')
        prob_part_new = partss[3].replace('MITRE techniques probability: ', '').strip().split(', ')    
        print("\nTechnique_testing_new\n")
        print(techniques_part_new)
        print("\nprob_testing_mew\n")
        print(prob_part_new)

        for technique_info in techniques_part_new:
            technique_name = technique_info.strip()
            techniques.append(technique_name)
            #print("DID THIS WORK??\n")
            #print(techniques)

        for prob_info in prob_part_new:
            probability = prob_info.split('(')[1].strip('%)').strip()
            probabilities.append(probability)

        for technique, probability in zip(techniques, probabilities):
            print(f"Class: {technique}, Probability: {float(probability) / 100:.1f}")
        
    except Exception as e:
        print(f"Error parsing GPT response: {e}")

    return techniques, probabilities

def process_classification(sentences, model_path, tokenizer_path, csv_file_path, label_column):
    
    print("Sentences received for classification:")
    print(sentences)

    predicted_labels, predicted_probabilities = classification(sentences, model_path, tokenizer_path)
    df = pd.read_csv(csv_file_path)
    unique_labels = df.iloc[:, label_column].unique()
    
    results = []
    for sentence, labelx, probabilities in zip(sentences, predicted_labels, predicted_probabilities):
        result = {
            'sentence': sentence,
            'predictions': []
        }
        sorted_indices = torch.argsort(probabilities, descending=True)[:3]
        for index in sorted_indices:
            label = unique_labels[index.item()]
            probability = probabilities[index].item()
            result['predictions'].append((label, probability))
        
        results.append(result)
    
    return results, unique_labels

def heuristic(results_model1, results_model2, gpt_results):
    final_results = []
    
    for i in range(len(results_model1)):
        techniques = defaultdict(list)

        # Collect all techniques and their probabilities from model A
        for label, prob in results_model1[i]['predictions']:
            techniques[label].append(prob)

        # Collect all techniques and their probabilities from model B
        for label, prob in results_model2[i]['predictions']:
            techniques[label].append(prob)
        
        # Collect all techniques and their probabilities from GPT results
        for label, prob in gpt_results[i]['predictions']:
            techniques[label].append(float(prob) / 100)

        # High Trust Agreement
        for tech, probs in techniques.items():
            if len(probs) >= 2 and any(p > 0.8 for p in probs):
                final_results.append((tech, max(probs)))
                break
        else:
            #$Top weighted Probability
            combined_probs = {}
            for tech, probs in techniques.items():
                combined_prob = 0
                for j, prob in enumerate(probs):
                    if j < 2:  # Model A and Model B
                        combined_prob += prob * 0.4
                    else:  # GPT model
                        combined_prob += prob * 0.2
                combined_probs[tech] = combined_prob

            best_tech, best_prob = max(combined_probs.items(), key=lambda x: x[1])
            if best_prob > 0.1:
                final_results.append((best_tech, best_prob))
            else:
                # Fallbackk
                gpt_probs = {label: float(prob) / 100 for label, prob in gpt_results[i]['predictions']}
                best_gpt_tech, best_gpt_prob = max(gpt_probs.items(), key=lambda x: x[1])
                final_results.append((best_gpt_tech, best_gpt_prob))

    return final_results

def display_results(sentences, results_model1, results_model2, chatbot_results, final_results):
    for i, sentence in enumerate(sentences):
        print("="*80)
        print(f"Sentence {i+1}: {sentence}")
        
        print("\nModel A Predictions (Trained on Sentences with keywords):")
        for label, prob in results_model1[i]['predictions']:
            print(f"Class: {label}, Probability: {float(prob):.2f}")
        
        print("\nModel B Predictions (Trained on Sentencea without keywords):")
        for label, prob in results_model2[i]['predictions']:
            print(f"Class: {label}, Probability: {float(prob):.2f}")
        
        print("\nGPT-4o Predictions:")
        for label, prob in chatbot_results[i]['predictions']:
            print(f"Class: {label}, Probability: {float(prob) / 100:.2f}")
        
        print("\nFinal Heuristic-based Prediction:")
        print(f"Class: {final_results[i][0]}, Probability: {final_results[i][1]:.2f}")
        
        print("="*80)

# Main processing
model_path_A = "Your trained model path here (for model trained on keywords)"
tokenizer_path_A = "Your tokenizer path for model A"
file_path_A = "Your csv file that was used to train model A"
label_column_A = 0

model_path_B = "Your trained model path here (for model trained on elaborated sentences)"
tokenizer_path_B = "Your tokenizer path for model B"
file_path_B = "Your csv file used to train model B"
label_column_B = 0

file_path = "Input file that contains unseen sentences for prediction.csv"
output_file = "TTPMapper_output.csv"

df_original = pd.read_csv(file_path)

df_original.dropna(subset=['Original Sentence'], inplace=True)

sentences_to_classify = df_original['Original Sentence'].tolist()
techniques_original = df_original['Technique'].tolist()

results_model1, unique_labels1 = process_classification(sentences_to_classify, model_path_A, tokenizer_path_A, file_path_A, label_column_A)
results_model2, unique_labels2 = process_classification(sentences_to_classify, model_path_B, tokenizer_path_B, file_path_B, label_column_B)


chatbot_results = []


for sentence in sentences_to_classify:
    gpt_response = query_gpt_api(sentence)
    if gpt_response:
        techniques, probabilities = parse_gpt_response(gpt_response)

        chatbot_results.append({
            'predictions': list(zip(techniques, probabilities)),
        })
    else:
        chatbot_results.append({
            
            'predictions': []     })
        
        
final_results = heuristic(results_model1, results_model2, chatbot_results)

output_data = []
for i in range(len(sentences_to_classify)):
        output_row = {
            'Original sentences': sentences_to_classify[i],
            'Techniques': techniques_original[i],
            'Model A Predictions': results_model1[i]['predictions'],
            'Model B Predictions': results_model2[i]['predictions'],
            'GPT-4o Predictions': chatbot_results[i]['predictions'],
            'Final Prediction': final_results[i]
        }
        output_data.append(output_row)

df_output = pd.DataFrame(output_data)
df_output.to_csv(output_file, index=False)



# Display all results
display_results(sentences_to_classify, results_model1, results_model2, chatbot_results, final_results)
