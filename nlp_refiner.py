from openai import OpenAI
import time
import torch
import requests

client = OpenAI(
    base_url="http://192.168.18.1:1234/v1",   # YOUR actual LM Studio server
    api_key="lm-studio"
)

def test_lm_studio_connection():
    """Test if LM Studio server is accessible"""
    try:
        response = requests.get("http://192.168.18.1:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return True, f"Connected! Available models: {len(models.get('data', []))}"
        else:
            return False, f"Server responded with status {response.status_code}"
    except requests.exceptions.ConnectRefused:
        return False, "Connection refused - LM Studio not running or wrong address"
    except requests.exceptions.Timeout:
        return False, "Connection timeout - check network/firewall"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def preprocess(chars):
    vowels = set("aeiou")
    result = []
    current = ""

    for c in chars:
        if len(current) >= 2 and all(ch not in vowels for ch in current[-2:]) and c not in vowels:
            result.append(current)
            current = c
        else:
            current += c

    if current:
        result.append(current)

    return " ".join(result)

def refine_buffer(buffer):
    """Legacy function name - kept for compatibility"""
    return refine_asl_buffer(buffer)

def refine_asl_buffer(buffer_input):
    """
    Refine ASL buffer input into proper English text
    
    Args:
        buffer_input: Either a string of letters (space-separated) or a list of characters
    
    Returns:
        dict: Contains refined_text, processing_time_seconds, model_device, preprocessed, cleaned
    """
    start_time = time.time()
    
    # Handle both string and list inputs
    if isinstance(buffer_input, str):
        # If it's a space-separated string, split it
        if ' ' in buffer_input:
            buffer = buffer_input.split()
        else:
            # If it's a continuous string, treat each character as separate
            buffer = list(buffer_input)
    else:
        # Assume it's already a list
        buffer = buffer_input
    
    # Clean the buffer - remove empty strings and convert to lowercase
    cleaned_buffer = [char.lower() for char in buffer if char and char.strip()]
    cleaned = ''.join(cleaned_buffer)
    
    # Preprocess
    preprocessed = preprocess(cleaned_buffer)
    
    # Test connection first
    connected, connection_msg = test_lm_studio_connection()
    if not connected:
        raise Exception(f"LM Studio connection failed: {connection_msg}")
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-0.5b-instruct",   # change to your loaded model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ASL buffer decoder.\n"
                        "Rules:\n"
                        "- You ARE allowed to add spaces between characters to form correct English words.\n"
                        "- Do not invent new words unless absolutely necessary.\n"
                        "- Do not add meaning that isn't implied.\n"
                        "- Only remove repeated characters when they don't belong.\n"
                        "- Output only the corrected English sentence.\n"
                        "- Keep it concise and natural.\n"
                    )
                },
                {
                    "role": "user",
                    "content": f"Decode this ASL letter sequence into proper English: {preprocessed}"
                }
            ],
            temperature=0.1
        )
        
        refined_text = response.choices[0].message.content.strip()
        
    except Exception as e:
        # Better error info for debugging
        raise Exception(f"LM Studio API error: {str(e)}")
    
    # Get device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processing_time = time.time() - start_time
    
    return {
        'refined_text': refined_text,
        'preprocessed': preprocessed,
        'cleaned': cleaned,
        'processing_time_seconds': round(processing_time, 3),
        'model_device': device,
        'connection_status': connection_msg
    }


# Test
if __name__ == "__main__":
    buffer = ['i','f','e','e','e','l','s','a','d','d']
    result = refine_asl_buffer(buffer)
    print("Original:", ''.join(buffer))
    print("Refined:", result['refined_text'])
    print("Processing time:", result['processing_time_seconds'], "seconds")
