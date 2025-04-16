
from dotenv import load_dotenv
import together
from together import Together

# Load environment variables
load_dotenv()

def completion(prompt, system_prompt=None):
    """
    Get completion from Together AI's Llama model
    
    Args:
        prompt (str): The prompt to send to the model
        system_prompt (str, optional): System prompt to prepend
    
    Returns:
        str: The model's completion
    """
    # Initialize the Together client
    client = together.Together()
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Create the completion using the chat API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
    )
    
    return response.choices[0].message.content