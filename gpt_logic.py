import google.generativeai as genai

# Directly set your Gemini API key
genai.configure(api_key="AIzaSyB3bbBOkjMH87QSfZx930etQ7DaC30VKdY")

# Load the Gemini model
model = genai.GenerativeModel("gemini-pro")

# Step 1: Infer the most likely prompt
def infer_prompt(text: str) -> str:
    response = model.generate_content(
        f"""You are a reverse prompt analyzer.
Given the following text, infer the most likely prompt that could have generated it.
The inferred prompt should be standalone, clear, and not reference the original text.

Text:
{text}
"""
    )
    return response.text.strip()

# Step 2: Generate 15 variations matching tone and length
def generate_variants(prompt: str, original_text: str) -> list[str]:
    outputs = []
    for i in range(15):
        response = model.generate_content(
            f"""Prompt:
{prompt}

Generate a unique, human-sounding response that matches the tone and length of the following example:
"{original_text}"
Avoid obvious AI structure or repetition.
"""
        )
        outputs.append(response.text.strip())
    return outputs

# Step 3: Humanize the text based on previously generated variants
def humanize_text(original_text: str, variants: list[str]) -> str:
    # Join all 15 outputs as context for rewriting
    joined_variants = "\n\n---\n\n".join(variants)
    
    response = model.generate_content(
        f"""Rewrite the following original text to sound more human and natural.
Avoid any repeating patterns, phrases, or tone found in these 15 generated variants.

Original:
{original_text}

Generated Variants:
{joined_variants}

Return only the final humanized version.
"""
    )
    return response.text.strip()
