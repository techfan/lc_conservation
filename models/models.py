from sqlalchemy import Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.sql import func
from db.database import Base


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class TableSchema(Base):
    __tablename__ = "table_schemas"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String, nullable=False, index=True)
    schema_info = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
