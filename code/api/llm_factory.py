import time

import vertexai
import vertexai.preview.generative_models as generative_models
from loguru import logger
from vertexai.generative_models import GenerativeModel

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


def generate(system_prompt, input_list):
    """Generate an output given a system prompt and an input list

    Args:
        system_prompt (str): the system prompt for the LLM
        input_list (list): a list of inputs for the LLM containing text and images(Part objects)

    Returns:
        dict: {"code": <response code (int)>, "response": <response text (str)>}
    """
    vertexai.init(project="kavach-440208", location="asia-south1")
    model = GenerativeModel("gemini-1.5-pro-002", system_instruction=[system_prompt])
    try:
        response = model.generate_content(
            input_list,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        output = response.text
        return {"code": 200, "response": output}

    except Exception as e:
        logger.warning(
            f"An error occured during response generation using the gemini-1.5-pro-002 model, error : {e}"
        )
        time.sleep(2)
        logger.warning("Sleeping for 60 seconds before trying again")
        time.sleep(60)

        try:
            response = model.generate_content(
                input_list,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            output = response.text
            return {"code": 200, "response": output}

        except Exception as e:
            return {
                "code": 512,
                "response": f"An error occured during response generation using the gemini-1.5-pro-002 model \n\n Error : {e}",
            }
