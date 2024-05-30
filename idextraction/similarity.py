#%%
from langchain.evaluation import load_evaluator
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate


import os
os.environ["ZHIPUAI_API_KEY"] = '8adac289dfd59cb2006ccea31a82efab.GhrrpvmaygKVqjxS'



def phrase_distance_hf(phrase1, phrase2):
    """ 
    similarity judgement using huggingface embeddings.
    input 2 phrases, return their similarity
    """
    embedding_model = HuggingFaceEmbeddings()
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_model)
    return evaluator.evaluate_string_pairs(prediction=phrase2, prediction_b=phrase1)['score']


def phrase_similarity_llm(phrase1, phrase2, chat_model):
    """
    similarity judgement using chat model
    input 2 phrases, return their similarity
    """
    ## TODO: add context? 
    response_schemas = [
        ResponseSchema(
            name="similarity", 
            type ="float", 
            description="similarity score of the two phrases, value within the range from 0.0 to 1.0."),
        ResponseSchema(
            name="synonymy",
            type="bool",
            description="judge if the two phrases have the same meaning in the context, output True if they are synonymy, False if not."
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    template = """\
        evaluate the similarity of the phrase '{phrase1}' and '{phrase2}', and judge whether they are synonymy in the context. 
        
        Output using the following JSON format: 
        
        {format_instructions}
        """

    prompt = PromptTemplate(
        template=template,
        input_variables=["phrase1", "phrase2"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | chat_model | output_parser
    return chain.invoke({"phrase1": phrase1, "phrase2": phrase2})


#%%
if __name__=='__main__':
    llm = ChatZhipuAI(
        temperature=0,
        model="glm-4",
        max_tokens=4096
    )

    p1 = "Opening within floors"
    p2 = "Openings connecting upper and lower floors within buildings"

    #print(phrase_distance_hf(p1, p2))
    print(phrase_similarity_llm(p1,p2,llm))
