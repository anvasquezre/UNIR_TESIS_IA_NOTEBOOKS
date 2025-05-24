from typing import List, Optional

from sqlalchemy.dialects.postgresql import ARRAY, VARCHAR
from sqlmodel import Field, Relationship, SQLModel


class Question(SQLModel, table=True):
    __tablename__ = "questions_questions"
    questionId: Optional[int] = Field(default=None, primary_key=True, unique=True)
    question: Optional[str] = Field(default=None)
    question_types: Optional[List[str]] = Field(default=None, sa_type=ARRAY(VARCHAR()))
    image_path: Optional[str] = Field(default=None)
    docId: Optional[int] = Field(default=None, foreign_key="questions_documents.docId")
    ucsf_document_id: Optional[str] = Field(default=None)
    ucsf_document_page_no: Optional[str] = Field(default=None)
    expected_answers: Optional[List[str]] = Field(
        default=None, sa_type=ARRAY(VARCHAR())
    )
    data_split: Optional[str] = Field(default=None)
    answers: Optional[List["Answer"]] = Relationship(
        back_populates="questions",
    )
    documents: Optional[List["Document"]] = Relationship(
        back_populates="questions",
    )


class Answer(SQLModel, table=True):
    __tablename__ = "question_answers"
    answerId: Optional[int] = Field(default=None, primary_key=True, unique=True)
    questionId: Optional[int] = Field(
        default=None, foreign_key="questions_questions.questionId"
    )
    answer: Optional[str] = Field(default=None)
    json_errors: Optional[int] = Field(default=None)
    experiment_label: Optional[str] = Field(default=None)
    questions: Optional[List[Question]] = Relationship(
        back_populates="answers",
    )


class Document(SQLModel, table=True):
    __tablename__ = "questions_documents"
    docId: Optional[int] = Field(default=None, primary_key=True, unique=True)
    ucsf_document_id: Optional[str] = Field(default=None)
    ucsf_document_page_no: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)
    llm_tokens_count: Optional[int] = Field(default=None)
    text_length: Optional[int] = Field(default=None)
    word_count: Optional[int] = Field(default=None)
    questions: Optional[List[Question]] = Relationship(
        back_populates="documents",
    )


target_metadata = SQLModel.metadata
