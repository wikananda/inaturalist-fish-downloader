"""Shared configuration constants for iNaturalist downloader tools."""

API_BASE = "https://api.inaturalist.org/v1"
DEFAULT_SPECIES_FILE = "species.txt"
DEFAULT_OUTPUT_DIR = "downloads"
DEFAULT_RAW_DIR = "downloads_raw"
DEFAULT_MANIFEST_DIR = "manifests"
DEFAULT_TIMEOUT = 30
IMAGES_PER_SPECIES = 30
DEFAULT_SPECIES_WORKERS = 5
DEFAULT_DOWNLOAD_WORKERS = 10
USER_AGENT = "inaturalist-downloader/1.0"
VALID_GRADES = {"any", "research", "needs_id", "casual"}
ALIVE_OR_DEAD_TERM_ID = 17
ALIVE_TERM_VALUE_ID = 18
