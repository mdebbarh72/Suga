from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    Glucose: float = Field(
        gt=0,
        le=500,
        description="mg/dL, typical range 70-200"
    )

    BloodPressure: float = Field(
        gt=0,
        le=250,
        description="mmHg, typical range 60-140"
    )

    SkinThickness: float = Field(
        ge=0,
        le=100,
        description="mm, typical range 10-50"
    )

    Insulin: float = Field(
        ge=0,
        le=1000,
        description="µU/mL, typical range 15-300"
    )

    BMI: float = Field(
        gt=0,
        le=100,
        description="kg/m², typical range 10-70"
    )

    DiabetesPedigreeFunction: float = Field(
        ge=0,
        le=3,
        description="Diabetes pedigree function"
    )

    Age: float = Field(
        gt=0,
        le=120,
        description="years"
    )