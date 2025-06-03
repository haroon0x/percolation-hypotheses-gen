from src.main import *

def generate_hypothesis(phenomenon, complexity, pdf_data):
    contents = [f"Phenomenon or topic to create a hypothesis on : {phenomenon}", f"Complexity value : {complexity}"]
    if not pdf_data:
        return False
    pdf = types.Part.from_bytes(
        data=pdf_data,
        mime_type='application/pdf',
    )
    contents.append(pdf)
    
    try:
        response = client.models.generate_content( model="gemini-2.0-flash", contents=contents ,config=GenerationConfig.gen_config_hyp_gen)
        return response.text.strip()
    except ServerError as e:
        print(f"⚠️ primary_model is overloaded. Switching to fallback_model...")
        try:
            response = client.models.generate_content( model="gemini-2.0-flash-lite", contents=contents ,config=GenerationConfig.gen_config_hyp_gen)
            return response.text.strip()
        except Exception as fallback_error:
            print("❌ Fallback model also failed:", fallback_error)
            return "All models are currently unavailable. Please try again later."
    except Exception as e:
        print("🔥 Unexpected error:", e)
        return "An unexpected error occurred during generation."

def analyze_complexity(hypothesis : str):
    """
    Returns the Complexity of scientific hypothesis in a single floating point number (1-100)
    """
    response = client.models.generate_content(model="gemini-2.0-flash", contents = hypothesis, config= GenerationConfig.gen_config_complexity_analyzer)
    return response.text

def get_info_density(hypothesis : str):
    """
    Returns the information density of the hypothesis on a scale of 1- 100
    """
    response = client_.models.generate_content(model="gemini-2.0-flash", contents = hypothesis, config= GenerationConfig.gen_config_info_density)
    return response.text


def generate_info_density(info):
    pass