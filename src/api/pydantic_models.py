# src/api/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    TotalTransactionAmount: float
    AvgTransactionAmount: float
    TransactionCount: float
    StdTransactionAmount: float
    CurrencyCode_UGX: float
    ProductCategory_airtime: float
    ProductCategory_data_bundles: float
    ProductCategory_financial_services: float
    ProductCategory_movies: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_transport: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ChannelId_ChannelId_1: float
    ChannelId_ChannelId_2: float
    ChannelId_ChannelId_3: float
    ChannelId_ChannelId_5: float

class PredictResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted fraud probability between 0 and 1.")
