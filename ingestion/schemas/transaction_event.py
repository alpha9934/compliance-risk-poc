from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from enum import Enum


class RiskClass(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TransactionEvent(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-2026-001",
                "customer_id": "CUST-42",
                "amount": 95000.00,
                "currency": "USD",
                "channel": "WIRE",
                "origin_account": "ACC-US-123",
                "destination_account": "ACC-AE-456",
                "beneficiary_name": "Gulf Trading LLC",
                "jurisdiction_origin": "US",
                "jurisdiction_destination": "AE",
                "product_type": "WIRE_TRANSFER",
                "timestamp": "2026-06-13T10:00:00Z"
            }
        }
    )

    transaction_id: str
    customer_id: str
    amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    channel: str                          # WIRE | ACH | SWIFT | CARD
    origin_account: str
    destination_account: str
    beneficiary_name: Optional[str] = None
    jurisdiction_origin: str = Field(..., min_length=2, max_length=2)
    jurisdiction_destination: str = Field(..., min_length=2, max_length=2)
    product_type: str
    timestamp: datetime
    schema_version: str = "1.0"
