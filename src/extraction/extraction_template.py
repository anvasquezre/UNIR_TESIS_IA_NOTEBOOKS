from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

template = ChatPromptTemplate(
    [
        (
            "system",
            "You have being tasked to answer the following question based on the provided context.",
        ),
        ("human", "Context: {context}. Question: {question}"),
        ("human", "Please provide the answer in the following format: {format}"),
        (
            "human",
            "Be concise and precise in your answer. Only provide the answer in the exact format it is provided in the context, do not add any additional information.",
        ),
        ("ai", "Answer:"),
    ]
)

Base64Image = str


def create_multimodal_template(
    image: Base64Image,
) -> ChatPromptTemplate:
    """Create a prompt template for the image."""

    image_template = ChatPromptTemplate(
        [
            (
                "system",
                "You have being tasked to answer the following question based on the provided context.",
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Here is an image of the document:"},
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image,
                        "mime_type": "image/jpeg",
                    },
                ],
            ),
            ("human", "The text extraction of that image is the following: {context}"),
            (
                "human",
                "Be concise and precise in your answer. Only provide the answer in the exact format it is provided in the context, do not add any additional information.",
            ),
            ("human", "Please provide the answer in the following format: {format}"),
            ("human", "Now answer the following question: {question}"),
            ("ai", "Answer:"),
        ]
    )
    return image_template


def create_image_only_template(
    image: Base64Image,
) -> ChatPromptTemplate:
    image_template = ChatPromptTemplate(
        [
            (
                "system",
                "You have being tasked to answer the following question based on the provided context.",
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Here is an image of the document:"},
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image,
                        "mime_type": "image/jpeg",
                    },
                ],
            ),
            (
                "human",
                "Be concise and precise in your answer. Only provide the answer in the exact format it is provided in the context, do not add any additional information.",
            ),
            ("human", "Please provide the answer in the following format: {format}"),
            ("human", "Now answer the following question: {question}"),
            ("ai", "Answer:"),
        ]
    )
    return image_template
