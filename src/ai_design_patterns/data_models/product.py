from enum import Enum

from pydantic import BaseModel, Field


class ProductCategory(Enum):
    laptops = "Laptops"
    mobile_phones = "Mobile Phones"
    desktops = "Desktops"
    tablets = "Tablets"
    accessories = "Accessories"


class Product(BaseModel):
    category: ProductCategory = Field(..., description="Category of the product")
    name: str = Field(..., description="Name of the product")
    cpu: str | None = Field(None, description="CPU of the product")
    ram: str | None = Field(None, description="RAM of the product")
    display: str | None = Field(None, description="Display of the product")
    gpu: str | None = Field(None, description="GPU of the product")
    storage: str | None = Field(None, description="Storage of the product")
    operating_system: str | None = Field(None, description="Operating system of the product")
    price: float = Field(..., description="Price of the product")
    features: list[str] = Field(default_factory=list, description="List of features of the product")