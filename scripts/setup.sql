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
--
-- Architecture: Uses Snowflake Cortex Model API run_batch() for distributed
-- inference instead of Ray clusters.
-- ============================================================================

USE ROLE ACCOUNTADMIN;

-- ============================================================================
-- SECTION 1: DATABASE AND SCHEMA (created first so cleanup can reference them)
-- ============================================================================
CREATE DATABASE IF NOT EXISTS SF_CLINICAL_DB
  COMMENT = 'Database for MONAI medical image processing solution';

CREATE SCHEMA IF NOT EXISTS SF_CLINICAL_DB.UTILS
  COMMENT = 'MONAI utilities: stages, models, and configurations';

-- ============================================================================
-- SECTION 2: CLEANUP FOR RE-RUNS (handles dependencies in correct order)
-- ============================================================================
-- Drop inference services first (created by run_batch()) before dropping model
-- Services have dynamic names like BATCH_INFERENCE_<UUID> and MODEL_BUILD_<UUID>
-- The model cannot be dropped while services reference it

-- Create a helper procedure to drop all services managed by a model
CREATE OR REPLACE PROCEDURE SF_CLINICAL_DB.UTILS.CLEANUP_MODEL_SERVICES(MODEL_FQN STRING)
RETURNS STRING
LANGUAGE JAVASCRIPT
EXECUTE AS CALLER
AS
$$
  var services = [];
  var stmt = snowflake.createStatement({sqlText: "SHOW SERVICES IN SCHEMA SF_CLINICAL_DB.UTILS"});
  stmt.execute();
  
  var result = snowflake.createStatement({
    sqlText: "SELECT \"name\" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) WHERE \"managing_object_name\" = ?",
    binds: [MODEL_FQN]
  });
  var rs = result.execute();
  
  while (rs.next()) {
    var svcName = rs.getColumnValue(1);
    try {
      snowflake.execute({sqlText: "DROP SERVICE IF EXISTS SF_CLINICAL_DB.UTILS." + svcName});
      services.push(svcName);
    } catch (err) {
      // Ignore errors for services that may already be gone
    }
  }
  return "Dropped " + services.length + " services: " + services.join(", ");
$$;

CALL SF_CLINICAL_DB.UTILS.CLEANUP_MODEL_SERVICES('SF_CLINICAL_DB.UTILS.LUNG_CT_REGISTRATION');
DROP PROCEDURE IF EXISTS SF_CLINICAL_DB.UTILS.CLEANUP_MODEL_SERVICES(STRING);

DROP MODEL IF EXISTS SF_CLINICAL_DB.UTILS.LUNG_CT_REGISTRATION;
DROP NOTEBOOK IF EXISTS SF_CLINICAL_DB.UTILS.SF_CLINICAL_01_INGEST_DATA;
DROP NOTEBOOK IF EXISTS SF_CLINICAL_DB.UTILS.SF_CLINICAL_02_MODEL_TRAINING;
DROP NOTEBOOK IF EXISTS SF_CLINICAL_DB.UTILS.SF_CLINICAL_03_MODEL_INFERENCE;
DROP ROLE IF EXISTS SF_CLINICAL_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 3: ROLE SETUP
-- ============================================================================
CREATE ROLE SF_CLINICAL_DATA_SCIENTIST
  COMMENT = 'Role for MONAI medical image processing notebooks';

SET my_user_var = (SELECT '"' || CURRENT_USER() || '"');
GRANT ROLE SF_CLINICAL_DATA_SCIENTIST TO USER identifier($my_user_var);

GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE SF_CLINICAL_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 4: WAREHOUSE SETUP
-- ============================================================================
CREATE OR REPLACE WAREHOUSE SF_CLINICAL_WH
  WAREHOUSE_SIZE = 'SMALL'
  WAREHOUSE_TYPE = 'STANDARD'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Warehouse for MONAI medical image processing';

GRANT USAGE ON WAREHOUSE SF_CLINICAL_WH TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT OPERATE ON WAREHOUSE SF_CLINICAL_WH TO ROLE SF_CLINICAL_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 5: SCHEMAS
-- ============================================================================
USE DATABASE SF_CLINICAL_DB;

CREATE OR REPLACE SCHEMA UTILS
  COMMENT = 'MONAI utilities: stages, models, and configurations';

CREATE OR REPLACE SCHEMA RESULTS
  COMMENT = 'MONAI inference results and metrics';

GRANT USAGE ON DATABASE SF_CLINICAL_DB TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT USAGE ON SCHEMA SF_CLINICAL_DB.UTILS TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT USAGE ON SCHEMA SF_CLINICAL_DB.RESULTS TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT ALL ON SCHEMA SF_CLINICAL_DB.UTILS TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT ALL ON SCHEMA SF_CLINICAL_DB.RESULTS TO ROLE SF_CLINICAL_DATA_SCIENTIST;

USE SCHEMA UTILS;

-- ============================================================================
-- SECTION 6: ENCRYPTED STAGES FOR MEDICAL IMAGES
-- ============================================================================
CREATE OR REPLACE STAGE SF_CLINICAL_MEDICAL_IMAGES_STG 
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

GRANT READ, WRITE ON STAGE SF_CLINICAL_DB.UTILS.SF_CLINICAL_MEDICAL_IMAGES_STG TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT READ, WRITE ON STAGE SF_CLINICAL_DB.UTILS.RESULTS_STG TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT READ, WRITE ON STAGE SF_CLINICAL_DB.UTILS.NOTEBOOK_STG TO ROLE SF_CLINICAL_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 7: NETWORK RULE AND EXTERNAL ACCESS INTEGRATION
-- ============================================================================
CREATE OR REPLACE NETWORK RULE ALLOW_ALL_NETWORK_RULES
  MODE = EGRESS 
  TYPE = HOST_PORT
  VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80', 'zenodo.org:443')
  COMMENT = 'Allow outbound HTTPS/HTTP for package installation';

CREATE OR REPLACE NETWORK RULE GITHUB_NETWORK_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('raw.githubusercontent.com:443')
  COMMENT = 'Allow access to GitHub for downloading notebooks';

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION SF_CLINICAL_ALLOW_ALL_EAI 
  ALLOWED_NETWORK_RULES = (SF_CLINICAL_DB.UTILS.ALLOW_ALL_NETWORK_RULES)
  ENABLED = TRUE
  COMMENT = 'External access for MONAI notebooks to install dependencies';

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION GITHUB_ACCESS_INTEGRATION
  ALLOWED_NETWORK_RULES = (SF_CLINICAL_DB.UTILS.GITHUB_NETWORK_RULE)
  ENABLED = TRUE
  COMMENT = 'External access to GitHub for downloading notebooks';

GRANT USAGE ON INTEGRATION SF_CLINICAL_ALLOW_ALL_EAI TO ROLE SF_CLINICAL_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 8: GPU COMPUTE POOL
-- ============================================================================
CREATE COMPUTE POOL IF NOT EXISTS SF_CLINICAL_GPU_ML_M_POOL 
  MIN_NODES = 1
  MAX_NODES = 8 
  INSTANCE_FAMILY = 'GPU_NV_M'
  COMMENT = 'GPU compute pool for MONAI medical image processing and run_batch() inference';

GRANT USAGE ON COMPUTE POOL SF_CLINICAL_GPU_ML_M_POOL TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT MONITOR ON COMPUTE POOL SF_CLINICAL_GPU_ML_M_POOL TO ROLE SF_CLINICAL_DATA_SCIENTIST;

-- ============================================================================
-- SECTION 9: NOTEBOOK AND MODEL PRIVILEGES
-- ============================================================================
GRANT CREATE NOTEBOOK ON SCHEMA SF_CLINICAL_DB.UTILS TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT CREATE MODEL ON SCHEMA SF_CLINICAL_DB.UTILS TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT CREATE TABLE ON SCHEMA SF_CLINICAL_DB.UTILS TO ROLE SF_CLINICAL_DATA_SCIENTIST;
GRANT CREATE TABLE ON SCHEMA SF_CLINICAL_DB.RESULTS TO ROLE SF_CLINICAL_DATA_SCIENTIST;

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
    stage_path = "@SF_CLINICAL_DB.UTILS.NOTEBOOK_STG"
    
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
USE WAREHOUSE SF_CLINICAL_WH;
CALL LOAD_NOTEBOOKS_FROM_GITHUB();

LIST @SF_CLINICAL_DB.UTILS.NOTEBOOK_STG;

-- ============================================================================
-- SECTION 12: CREATE NOTEBOOKS FROM STAGE
-- ============================================================================
CREATE OR REPLACE NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_01_INGEST_DATA
    FROM '@SF_CLINICAL_DB.UTILS.NOTEBOOK_STG'
    MAIN_FILE = '01_ingest_data.ipynb'
    QUERY_WAREHOUSE = 'SF_CLINICAL_WH'
    COMPUTE_POOL = 'SF_CLINICAL_GPU_ML_M_POOL'
    RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
    COMMENT = 'MONAI Data Ingestion - Downloads and uploads lung CT scans';

CREATE OR REPLACE NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_02_MODEL_TRAINING
    FROM '@SF_CLINICAL_DB.UTILS.NOTEBOOK_STG'
    MAIN_FILE = '02_model_training.ipynb'
    QUERY_WAREHOUSE = 'SF_CLINICAL_WH'
    COMPUTE_POOL = 'SF_CLINICAL_GPU_ML_M_POOL'
    RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
    COMMENT = 'MONAI Model Training - Trains LocalNet registration model with @remote decorator';

CREATE OR REPLACE NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_03_MODEL_INFERENCE
    FROM '@SF_CLINICAL_DB.UTILS.NOTEBOOK_STG'
    MAIN_FILE = '03_model_inference.ipynb'
    QUERY_WAREHOUSE = 'SF_CLINICAL_WH'
    COMPUTE_POOL = 'SF_CLINICAL_GPU_ML_M_POOL'
    RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
    COMMENT = 'MONAI Model Inference - Distributed inference using Cortex Model API run_batch()';

-- ============================================================================
-- SECTION 13: CREATE LIVE VERSIONS (required to open notebooks in UI)
-- ============================================================================
ALTER NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_01_INGEST_DATA ADD LIVE VERSION FROM LAST;
ALTER NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_02_MODEL_TRAINING ADD LIVE VERSION FROM LAST;
ALTER NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_03_MODEL_INFERENCE ADD LIVE VERSION FROM LAST;

-- ============================================================================
-- SECTION 14: CONFIGURE EXTERNAL ACCESS ON NOTEBOOKS
-- ============================================================================
ALTER NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_01_INGEST_DATA 
    SET EXTERNAL_ACCESS_INTEGRATIONS = (SF_CLINICAL_ALLOW_ALL_EAI);

ALTER NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_02_MODEL_TRAINING 
    SET EXTERNAL_ACCESS_INTEGRATIONS = (SF_CLINICAL_ALLOW_ALL_EAI);

ALTER NOTEBOOK SF_CLINICAL_DB.UTILS.SF_CLINICAL_03_MODEL_INFERENCE 
    SET EXTERNAL_ACCESS_INTEGRATIONS = (SF_CLINICAL_ALLOW_ALL_EAI);

-- ============================================================================
-- SECTION 15: VERIFICATION
-- ============================================================================
SELECT 'Setup complete! Verify objects below:' AS status;
SHOW SCHEMAS IN DATABASE SF_CLINICAL_DB;
SHOW STAGES IN SCHEMA SF_CLINICAL_DB.UTILS;
SHOW NOTEBOOKS IN SCHEMA SF_CLINICAL_DB.UTILS;
SHOW COMPUTE POOLS LIKE '%SF_CLINICAL%';

USE ROLE SF_CLINICAL_DATA_SCIENTIST;
USE WAREHOUSE SF_CLINICAL_WH;
USE DATABASE SF_CLINICAL_DB;
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
--    - SF_CLINICAL_01_INGEST_DATA
--    - SF_CLINICAL_02_MODEL_TRAINING
--    - SF_CLINICAL_03_MODEL_INFERENCE
--
-- 3. Open each notebook and click "Start" to initialize the container runtime
--
-- 4. Run notebooks in order: 01 → 02 → 03
--
-- Architecture Notes:
-- - Notebook 02 uses @remote decorator to train on GPU compute pool
-- - Notebook 03 uses Cortex Model API run_batch() for distributed inference
--   (replaces Ray-based distributed processing)
-- - Model is registered with CustomModel class supporting InputSpec/OutputSpec
--
-- Note: The first run may take 2-3 minutes to start as the container initializes
-- ============================================================================

-- ============================================================================
-- SECTION 15: TEARDOWN SCRIPT (Uncomment to clean up all resources)
-- ============================================================================
-- IMPORTANT: Run in this exact order to handle dependencies correctly.
-- Models create internal pipes that block role deletion.

-- USE ROLE ACCOUNTADMIN;
-- DROP SERVICE IF EXISTS SF_CLINICAL_DB.UTILS.LUNG_CT_REGISTRATION_V_RUN_BATCH_8_SERVICE;
-- DROP SERVICE IF EXISTS SF_CLINICAL_DB.UTILS.LUNG_CT_REGISTRATION_V1_SERVICE;
-- DROP MODEL IF EXISTS SF_CLINICAL_DB.UTILS.LUNG_CT_REGISTRATION;
-- DROP NOTEBOOK IF EXISTS SF_CLINICAL_DB.UTILS.SF_CLINICAL_01_INGEST_DATA;
-- DROP NOTEBOOK IF EXISTS SF_CLINICAL_DB.UTILS.SF_CLINICAL_02_MODEL_TRAINING;
-- DROP NOTEBOOK IF EXISTS SF_CLINICAL_DB.UTILS.SF_CLINICAL_03_MODEL_INFERENCE;
-- DROP COMPUTE POOL IF EXISTS SF_CLINICAL_GPU_ML_M_POOL;
-- DROP EXTERNAL ACCESS INTEGRATION IF EXISTS SF_CLINICAL_ALLOW_ALL_EAI;
-- DROP EXTERNAL ACCESS INTEGRATION IF EXISTS GITHUB_ACCESS_INTEGRATION;
-- DROP DATABASE IF EXISTS SF_CLINICAL_DB CASCADE;
-- DROP WAREHOUSE IF EXISTS SF_CLINICAL_WH;
-- DROP ROLE IF EXISTS SF_CLINICAL_DATA_SCIENTIST;
