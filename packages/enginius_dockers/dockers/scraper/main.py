import logging
import sys
import traceback
from enum import Enum
from typing import List, Optional

from database import CONFIG_DIR
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Logging Configuration ---
# Configure root logger to output to stdout/stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Logging ---
logger = logging.getLogger(__name__)

# Import all scraper modules
from scraping import (
    AC_web,
    Additional_510k,
    AUS_Guidance,
    AUS_Regulation,
    Canada_Regulation,
    CFR_FAA,
    EASA,
    EU_MDCG,
    EU_MDR_Ammendment,
    EU_MDR,
    FDA_CFR,
    FDA_Guidance,
    NIST_RMF,
)

# --- Database Configuration ---
DB_CONFIG = parseCredentialFile(str(CONFIG_DIR / "dev_test_tlp_config.json"))

if DB_CONFIG is None:
    logger.error("Failed to load database configuration")
    raise RuntimeError("Database configuration not loaded")


# --- Scraper Enum ---
class Scraper(str, Enum):
    AC_WEB = "AC_web"
    ADDITIONAL_510K = "Additional_510k"
    AUS_GUIDANCE = "AUS_Guidance"
    AUS_REGULATION = "AUS_Regulation"
    CANADA_REGULATION = "Canada_Regulation"
    CFR_FAA = "CFR_FAA"
    EASA = "EASA"
    EU_MDCG = "EU_MDCG"
    EU_MDR_AMMENDMENT = "EU_MDR_Ammendment"
    EU_MDR = "EU_MDR"
    FDA_CFR = "FDA_CFR"
    FDA_GUIDANCE = "FDA_Guidance"
    NIST_RMF = "NIST_RMF"


# Map enum values to module objects
SCRAPER_MODULES = {
    Scraper.AC_WEB: AC_web,
    Scraper.ADDITIONAL_510K: Additional_510k,
    Scraper.AUS_GUIDANCE: AUS_Guidance,
    Scraper.AUS_REGULATION: AUS_Regulation,
    Scraper.CANADA_REGULATION: Canada_Regulation,
    Scraper.CFR_FAA: CFR_FAA,
    Scraper.EASA: EASA,
    Scraper.EU_MDCG: EU_MDCG,
    Scraper.EU_MDR_AMMENDMENT: EU_MDR_Ammendment,
    Scraper.EU_MDR: EU_MDR,
    Scraper.FDA_CFR: FDA_CFR,
    Scraper.FDA_GUIDANCE: FDA_Guidance,
    Scraper.NIST_RMF: NIST_RMF,
}


# --- FastAPI App ---
app = FastAPI(title="Scraper Service API")


# --- Pydantic Models ---
class RunScraperRequest(BaseModel):
    scraper: Scraper
    scrapeURLId: Optional[int] = None


class RunScraperResponse(BaseModel):
    success: bool
    message: str


# --- Endpoints ---
@app.get("/scrapers", response_model=List[str])
async def list_scrapers():
    """Returns list of available scrapers defined in the enum."""
    return [scraper.value for scraper in Scraper]


@app.post("/run", response_model=RunScraperResponse)
async def run_scraper(request: RunScraperRequest):
    """
    Executes a scraper's run() method.
    
    Args:
        request: Contains scraper name and optional scrapeURLId
        
    Returns:
        Success status and message
    """
    try:
        scraper_module = SCRAPER_MODULES.get(request.scraper)
        if scraper_module is None:
            raise HTTPException(
                status_code=404, message=f"Scraper {request.scraper.value} not found"
            )

        logger.info(f"Running scraper: {request.scraper.value}")
        
        # Other scrapers require scrapeURLId
        if request.scrapeURLId is None:
            raise HTTPException(
                status_code=400,
                message=f"scrapeURLId is required for scraper {request.scraper.value}",
            )
        logger.info(f"Running {request.scraper.value} with scrapeURLId: {request.scrapeURLId}")
        scraper_module.run(DB_CONFIG, request.scrapeURLId)

        return RunScraperResponse(
            success=True, message=f"Scraper {request.scraper.value} completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running scraper {request.scraper.value}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            message=f"Error running scraper {request.scraper.value}: {str(e)}",
        )

