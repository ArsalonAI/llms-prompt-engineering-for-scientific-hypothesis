from dotenv import load_dotenv
import together
# It's good practice to initialize the client once if possible,
# but for this standalone function, initializing per call is fine.
# client = together.Together() # Consider initializing globally if used frequently in a larger app

# Load environment variables
load_dotenv()

def llama_3_3_70B_completion(
    prompt, 
    system_prompt=None,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    max_tokens: int = 1024
    ):
    """
    Get completion from Together AI's Llama model with configurable hyperparameters.
    
    Args:
        prompt (str): The prompt to send to the model
        system_prompt (str, optional): System prompt to prepend
        temperature (float): Controls randomness. Higher is more creative.
        top_p (float): Nucleus sampling. Considers a smaller set of high-probability words.
        top_k (int): Considers top k most likely words.
        repetition_penalty (float): Penalizes repeated tokens.
        max_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        str: The model's completion
    """
    # Initialize the Together client within the function to ensure it's always available
    # and uses the latest environment variables if they were to change.
    client = together.Together()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo", # Ensure this model name is correct and available
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    
    return response.choices[0].message.content