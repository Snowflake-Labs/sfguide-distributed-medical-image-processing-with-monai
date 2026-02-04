-- Copyright 2026 Snowflake Inc.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- ============================================================================
-- Distributed Medical Image Processing with MONAI - Setup Script
-- ============================================================================
-- This script creates all required Snowflake objects and automatically
-- downloads and configures the notebooks from GitHub.
-- ============================================================================

USE ROLE ACCOUNTADMIN;

-- ============================================================================
-- SECTION 1: DATABASE AND SCHEMA (created first so cleanup can reference them)
-- ============================================================================
CREATE DATABASE IF NOT EXISTS MONAI_DB
  COMMENT = 'Database for MONAI medical image processing solution';

CREATE SCHEMA IF NOT EXISTS MONAI_DB.UTILS
  COMMENT = 'MONAI utilities: stages, models, and configurations';

-- ============================================================================
-- SECTION 2: CLEANUP FOR RE-RUNS (handles dependencies in correct order)
-- ============================================================================
-- Models must be dropped BEFORE the role, as models create internal pipes
-- that prevent role deletion
DROP MODEL IF EXISTS MONAI_DB.UTILS.LUNG_CT_REGISTRATION;
DROP NOTEBOOK IF EXISTS MONAI_DB.UTILS.MONAI_01_INGEST_DATA;
DROP NOTEBOOK IF EXISTS MONAI_DB.UTILS.MONAI_02_MODEL_TRAINING;
DROP NOTEBOOK IF EXISTS MONAI_DB.UTILS.MONAI_03_MODEL_INFERENCE;
DROP ROLE IF EXISTS MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 3: ROLE SETUP
-- ============================================================================
CREATE ROLE MONAI_DATA_SCIENTIST
  COMMENT = 'Role for MONAI medical image processing notebooks';

-- Grant role to current user
SET my_user_var = (SELECT '"' || CURRENT_USER() || '"');
GRANT ROLE MONAI_DATA_SCIENTIST TO USER identifier($my_user_var);

-- Grant Cortex privileges (required for AI/ML functions)
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 4: WAREHOUSE SETUP
-- ============================================================================
CREATE OR REPLACE WAREHOUSE MONAI_WH
  WAREHOUSE_SIZE = 'SMALL'
  WAREHOUSE_TYPE = 'STANDARD'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Warehouse for MONAI medical image processing';

GRANT USAGE ON WAREHOUSE MONAI_WH TO ROLE MONAI_DATA_SCIENTIST;
GRANT OPERATE ON WAREHOUSE MONAI_WH TO ROLE MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 5: SCHEMAS
-- ============================================================================
USE DATABASE MONAI_DB;

CREATE OR REPLACE SCHEMA UTILS
  COMMENT = 'MONAI utilities: stages, models, and configurations';

CREATE OR REPLACE SCHEMA RESULTS
  COMMENT = 'MONAI inference results and metrics';

-- Grant database and schema access
GRANT USAGE ON DATABASE MONAI_DB TO ROLE MONAI_DATA_SCIENTIST;
GRANT USAGE ON SCHEMA MONAI_DB.UTILS TO ROLE MONAI_DATA_SCIENTIST;
GRANT USAGE ON SCHEMA MONAI_DB.RESULTS TO ROLE MONAI_DATA_SCIENTIST;
GRANT ALL ON SCHEMA MONAI_DB.UTILS TO ROLE MONAI_DATA_SCIENTIST;
GRANT ALL ON SCHEMA MONAI_DB.RESULTS TO ROLE MONAI_DATA_SCIENTIST;

USE SCHEMA UTILS;

-- ============================================================================
-- SECTION 6: ENCRYPTED STAGES FOR MEDICAL IMAGES
-- ============================================================================
CREATE OR REPLACE STAGE MONAI_MEDICAL_IMAGES_STG 
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Lung CT scans and segmentation masks in NIfTI format';

CREATE OR REPLACE STAGE RESULTS_STG 
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Registered images, model checkpoints, and inference outputs';

CREATE OR REPLACE STAGE NOTEBOOK_STG
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Stage for notebook files';

-- Grant stage access
GRANT READ, WRITE ON STAGE MONAI_DB.UTILS.MONAI_MEDICAL_IMAGES_STG TO ROLE MONAI_DATA_SCIENTIST;
GRANT READ, WRITE ON STAGE MONAI_DB.UTILS.RESULTS_STG TO ROLE MONAI_DATA_SCIENTIST;
GRANT READ, WRITE ON STAGE MONAI_DB.UTILS.NOTEBOOK_STG TO ROLE MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 7: NETWORK RULE AND EXTERNAL ACCESS INTEGRATION
-- ============================================================================
CREATE OR REPLACE NETWORK RULE ALLOW_ALL_NETWORK_RULES
  MODE = EGRESS 
  TYPE = HOST_PORT
  VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80')
  COMMENT = 'Allow outbound HTTPS/HTTP for package installation';

CREATE OR REPLACE NETWORK RULE GITHUB_NETWORK_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('raw.githubusercontent.com:443')
  COMMENT = 'Allow access to GitHub for downloading notebooks';

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION MONAI_ALLOW_ALL_EAI 
  ALLOWED_NETWORK_RULES = (MONAI_DB.UTILS.ALLOW_ALL_NETWORK_RULES)
  ENABLED = TRUE
  COMMENT = 'External access for MONAI notebooks to install dependencies';

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION GITHUB_ACCESS_INTEGRATION
  ALLOWED_NETWORK_RULES = (MONAI_DB.UTILS.GITHUB_NETWORK_RULE)
  ENABLED = TRUE
  COMMENT = 'External access to GitHub for downloading notebooks';

GRANT USAGE ON INTEGRATION MONAI_ALLOW_ALL_EAI TO ROLE MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 8: GPU COMPUTE POOL
-- ============================================================================
CREATE COMPUTE POOL IF NOT EXISTS MONAI_GPU_ML_M_POOL 
  MIN_NODES = 1
  MAX_NODES = 8 
  INSTANCE_FAMILY = 'GPU_NV_M'
  COMMENT = 'GPU compute pool for MONAI medical image processing';

GRANT USAGE ON COMPUTE POOL MONAI_GPU_ML_M_POOL TO ROLE MONAI_DATA_SCIENTIST;
GRANT MONITOR ON COMPUTE POOL MONAI_GPU_ML_M_POOL TO ROLE MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 9: NOTEBOOK PRIVILEGES
-- ============================================================================
GRANT CREATE NOTEBOOK ON SCHEMA MONAI_DB.UTILS TO ROLE MONAI_DATA_SCIENTIST;
GRANT CREATE MODEL ON SCHEMA MONAI_DB.UTILS TO ROLE MONAI_DATA_SCIENTIST;
GRANT CREATE TABLE ON SCHEMA MONAI_DB.UTILS TO ROLE MONAI_DATA_SCIENTIST;
GRANT CREATE TABLE ON SCHEMA MONAI_DB.RESULTS TO ROLE MONAI_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 10: GITHUB NOTEBOOK LOADER PROCEDURE
-- ============================================================================
CREATE OR REPLACE PROCEDURE LOAD_NOTEBOOKS_FROM_GITHUB()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'requests')
HANDLER = 'load_notebooks'
EXTERNAL_ACCESS_INTEGRATIONS = (GITHUB_ACCESS_INTEGRATION)
AS
$$
import requests
from snowflake.snowpark import Session

def load_notebooks(session: Session) -> str:
    base_url = "https://raw.githubusercontent.com/Snowflake-Labs/sfguide-distributed-medical-image-processing-with-monai/main/notebooks"
    stage_path = "@MONAI_DB.UTILS.NOTEBOOK_STG"
    
    notebooks = [
        "01_ingest_data.ipynb",
        "02_model_training.ipynb",
        "03_model_inference.ipynb"
    ]
    
    results = []
    
    for notebook in notebooks:
        try:
            url = f"{base_url}/{notebook}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            local_path = f"/tmp/{notebook}"
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            session.file.put(local_path, stage_path, auto_compress=False, overwrite=True)
            results.append(f"✓ {notebook}: uploaded to {stage_path}")
            
        except Exception as e:
            results.append(f"✗ {notebook}: Error - {str(e)}")
    
    return "\n".join(results)
$$;

-- ============================================================================
-- SECTION 11: DOWNLOAD NOTEBOOKS FROM GITHUB
-- ============================================================================
USE WAREHOUSE MONAI_WH;
CALL LOAD_NOTEBOOKS_FROM_GITHUB();

-- Verify notebooks uploaded
LIST @MONAI_DB.UTILS.NOTEBOOK_STG;

-- ============================================================================
-- SECTION 12: CREATE NOTEBOOKS FROM STAGE
-- ============================================================================
CREATE OR REPLACE NOTEBOOK MONAI_DB.UTILS.MONAI_01_INGEST_DATA
    FROM '@MONAI_DB.UTILS.NOTEBOOK_STG'
    MAIN_FILE = '01_ingest_data.ipynb'
    QUERY_WAREHOUSE = 'MONAI_WH'
    COMPUTE_POOL = 'MONAI_GPU_ML_M_POOL'
    RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
    COMMENT = 'MONAI Data Ingestion - Downloads and uploads lung CT scans';

CREATE OR REPLACE NOTEBOOK MONAI_DB.UTILS.MONAI_02_MODEL_TRAINING
    FROM '@MONAI_DB.UTILS.NOTEBOOK_STG'
    MAIN_FILE = '02_model_training.ipynb'
    QUERY_WAREHOUSE = 'MONAI_WH'
    COMPUTE_POOL = 'MONAI_GPU_ML_M_POOL'
    RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
    COMMENT = 'MONAI Model Training - Trains LocalNet registration model';

CREATE OR REPLACE NOTEBOOK MONAI_DB.UTILS.MONAI_03_MODEL_INFERENCE
    FROM '@MONAI_DB.UTILS.NOTEBOOK_STG'
    MAIN_FILE = '03_model_inference.ipynb'
    QUERY_WAREHOUSE = 'MONAI_WH'
    COMPUTE_POOL = 'MONAI_GPU_ML_M_POOL'
    RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
    COMMENT = 'MONAI Model Inference - Runs distributed inference';

-- ============================================================================
-- SECTION 13: CONFIGURE EXTERNAL ACCESS ON NOTEBOOKS
-- ============================================================================
ALTER NOTEBOOK MONAI_DB.UTILS.MONAI_01_INGEST_DATA 
    SET EXTERNAL_ACCESS_INTEGRATIONS = (MONAI_ALLOW_ALL_EAI);

ALTER NOTEBOOK MONAI_DB.UTILS.MONAI_02_MODEL_TRAINING 
    SET EXTERNAL_ACCESS_INTEGRATIONS = (MONAI_ALLOW_ALL_EAI);

ALTER NOTEBOOK MONAI_DB.UTILS.MONAI_03_MODEL_INFERENCE 
    SET EXTERNAL_ACCESS_INTEGRATIONS = (MONAI_ALLOW_ALL_EAI);

-- ============================================================================
-- SECTION 14: VERIFICATION
-- ============================================================================
SELECT 'Setup complete! Verify objects below:' AS status;
SHOW SCHEMAS IN DATABASE MONAI_DB;
SHOW STAGES IN SCHEMA MONAI_DB.UTILS;
SHOW NOTEBOOKS IN SCHEMA MONAI_DB.UTILS;
SHOW COMPUTE POOLS LIKE '%MONAI%';

-- Switch to the data scientist role for notebook work
USE ROLE MONAI_DATA_SCIENTIST;
USE WAREHOUSE MONAI_WH;
USE DATABASE MONAI_DB;
USE SCHEMA UTILS;

SELECT 'Infrastructure setup complete!' AS status, 
       CURRENT_ROLE() AS current_role, 
       CURRENT_WAREHOUSE() AS current_warehouse,
       CURRENT_DATABASE() AS current_database;

-- ============================================================================
-- NEXT STEPS: RUN THE NOTEBOOKS
-- ============================================================================
-- 1. In Snowsight, navigate to Projects → Notebooks
--
-- 2. You should see three notebooks already created:
--    - MONAI_01_INGEST_DATA
--    - MONAI_02_MODEL_TRAINING
--    - MONAI_03_MODEL_INFERENCE
--
-- 3. Open each notebook and click "Start" to initialize the container runtime
--
-- 4. Run notebooks in order: 01 → 02 → 03
--
-- Note: The first run may take 2-3 minutes to start as the container initializes
-- ============================================================================

-- ============================================================================
-- SECTION 15: TEARDOWN SCRIPT (Uncomment to clean up all resources)
-- ============================================================================
-- IMPORTANT: Run in this exact order to handle dependencies correctly.
-- Models create internal pipes that block role deletion.

-- USE ROLE ACCOUNTADMIN;
-- DROP MODEL IF EXISTS MONAI_DB.UTILS.LUNG_CT_REGISTRATION;
-- DROP NOTEBOOK IF EXISTS MONAI_DB.UTILS.MONAI_01_INGEST_DATA;
-- DROP NOTEBOOK IF EXISTS MONAI_DB.UTILS.MONAI_02_MODEL_TRAINING;
-- DROP NOTEBOOK IF EXISTS MONAI_DB.UTILS.MONAI_03_MODEL_INFERENCE;
-- DROP COMPUTE POOL IF EXISTS MONAI_GPU_ML_M_POOL;
-- DROP EXTERNAL ACCESS INTEGRATION IF EXISTS MONAI_ALLOW_ALL_EAI;
-- DROP EXTERNAL ACCESS INTEGRATION IF EXISTS GITHUB_ACCESS_INTEGRATION;
-- DROP DATABASE IF EXISTS MONAI_DB CASCADE;
-- DROP WAREHOUSE IF EXISTS MONAI_WH;
-- DROP ROLE IF EXISTS MONAI_DATA_SCIENTIST;
