import google.generativeai as genai

genai.configure(api_key="AIzaSyB3bbBOkjMH87QSfZx930etQ7DaC30VKdY")

model = genai.GenerativeModel("gemini-1.5-flash")  # This works in Google AI Studio

def safe_generate(prompt: str) -> str:
    try:
        print("🧠 Prompt to Gemini:")
        print(prompt)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini Error:", str(e))
        return "[ERROR] Gemini failed"

def infer_prompt(text: str) -> str:
    return safe_generate(
        f"""You are a reverse prompt analyzer.
Given the following text, infer the most likely prompt that could have generated it.
The inferred prompt should be standalone, clear, and not reference the original text.

Text:
{text}
"""
    )

def generate_variants(prompt: str, original_text: str) -> list[str]:
    outputs = []
    for i in range(15):
        output = safe_generate(
            f"""Prompt: {prompt}

Generate a humanlike response matching the tone and length of the following:
"{original_text}"

Avoid repetition and obvious AI writing patterns.
"""
        )
        outputs.append(output)
    return outputs

def humanize_text(original_text: str, variants: list[str]) -> str:
    joined_variants = "\n\n---\n\n".join(variants)
    return safe_generate(
        f"""Rewrite the original text to sound natural and humanlike.
Avoid stylistic patterns or phrasing found in the following generated variants.

Original:
{original_text}

Generated Variants:
{joined_variants}

Return only the final humanized version.
"""
    )
