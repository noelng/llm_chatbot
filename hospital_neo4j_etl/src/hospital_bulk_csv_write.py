import os
import logging
from retry import retry
from neo4j import GraphDatabase

# =========================
# Environment variables
# =========================

HOSPITALS_CSV_PATH = os.getenv("HOSPITALS_CSV_PATH")
PAYERS_CSV_PATH = os.getenv("PAYERS_CSV_PATH")
PHYSICIANS_CSV_PATH = os.getenv("PHYSICIANS_CSV_PATH")
PATIENTS_CSV_PATH = os.getenv("PATIENTS_CSV_PATH")
VISITS_CSV_PATH = os.getenv("VISITS_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")
EXAMPLE_CYPHER_CSV_PATH = os.getenv("EXAMPLE_CYPHER_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

# =========================
# Schema
# =========================

NODE_CONSTRAINTS = {
    "Hospital": "id",
    "Payer": "id",
    "Physician": "id",
    "Patient": "id",
    "Visit": "id",
    "Review": "id",
    "Question": "question",  # FIXED: Question has no `id`
}

# =========================
# Driver lifecycle
# =========================

def get_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

# =========================
# Retry ONLY DB operations
# =========================

@retry(tries=100, delay=10)
def run_query(driver, query):
    with driver.session(database="neo4j") as session:
        session.run(query)

# =========================
# ETL
# =========================

def load_hospital_graph_from_csv(driver) -> None:
    LOGGER.info("Setting uniqueness constraints")

    for label, field in NODE_CONSTRAINTS.items():
        query = f"""
        CREATE CONSTRAINT IF NOT EXISTS
        FOR (n:{label})
        REQUIRE n.{field} IS UNIQUE
        """
        run_query(driver, query)

    LOGGER.info("Loading Hospital nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{HOSPITALS_CSV_PATH}' AS h
    MERGE (:Hospital {{
        id: toInteger(h.hospital_id),
        name: h.hospital_name,
        state_name: h.hospital_state
    }})
    """)

    LOGGER.info("Loading Payer nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{PAYERS_CSV_PATH}' AS p
    MERGE (:Payer {{
        id: toInteger(p.payer_id),
        name: p.payer_name
    }})
    """)

    LOGGER.info("Loading Physician nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{PHYSICIANS_CSV_PATH}' AS p
    MERGE (:Physician {{
        id: toInteger(p.physician_id),
        name: p.physician_name,
        dob: p.physician_dob,
        grad_year: p.physician_grad_year,
        school: p.medical_school,
        salary: toFloat(p.salary)
    }})
    """)

    LOGGER.info("Loading Patient nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{PATIENTS_CSV_PATH}' AS p
    MERGE (:Patient {{
        id: toInteger(p.patient_id),
        name: p.patient_name,
        sex: p.patient_sex,
        dob: p.patient_dob,
        blood_type: p.patient_blood_type
    }})
    """)

    LOGGER.info("Loading Visit nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS v
    MERGE (visit:Visit {{
        id: toInteger(v.visit_id)
    }})
    SET visit.room_number = toInteger(v.room_number),
        visit.admission_type = v.admission_type,
        visit.admission_date = v.date_of_admission,
        visit.test_results = v.test_results,
        visit.status = v.visit_status,
        visit.chief_complaint = v.chief_complaint,
        visit.treatment_description = v.treatment_description,
        visit.diagnosis = v.primary_diagnosis,
        visit.discharge_date = v.discharge_date
    """)

    LOGGER.info("Loading Review nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{REVIEWS_CSV_PATH}' AS r
    MERGE (:Review {{
        id: toInteger(r.review_id),
        text: r.review,
        patient_name: r.patient_name,
        physician_name: r.physician_name,
        hospital_name: r.hospital_name
    }})
    """)

    LOGGER.info("Loading Question nodes")
    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{EXAMPLE_CYPHER_CSV_PATH}' AS q
    MERGE (:Question {{
        question: q.question,
        cypher: q.cypher
    }})
    """)

    LOGGER.info("Loading relationships")

    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS v
    MATCH (visit:Visit {{id: toInteger(v.visit_id)}})
    MATCH (hospital:Hospital {{id: toInteger(v.hospital_id)}})
    MERGE (visit)-[:AT]->(hospital)
    """)

    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{REVIEWS_CSV_PATH}' AS r
    MATCH (visit:Visit {{id: toInteger(r.visit_id)}})
    MATCH (review:Review {{id: toInteger(r.review_id)}})
    MERGE (visit)-[:WRITES]->(review)
    """)

    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS v
    MATCH (physician:Physician {{id: toInteger(v.physician_id)}})
    MATCH (visit:Visit {{id: toInteger(v.visit_id)}})
    MERGE (physician)-[:TREATS]->(visit)
    """)

    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS v
    MATCH (visit:Visit {{id: toInteger(v.visit_id)}})
    MATCH (payer:Payer {{id: toInteger(v.payer_id)}})
    MERGE (visit)-[c:COVERED_BY]->(payer)
    SET c.service_date = v.discharge_date,
        c.billing_amount = toFloat(v.billing_amount)
    """)

    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS v
    MATCH (patient:Patient {{id: toInteger(v.patient_id)}})
    MATCH (visit:Visit {{id: toInteger(v.visit_id)}})
    MERGE (patient)-[:HAS]->(visit)
    """)

    run_query(driver, f"""
    LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS v
    MATCH (hospital:Hospital {{id: toInteger(v.hospital_id)}})
    MATCH (physician:Physician {{id: toInteger(v.physician_id)}})
    MERGE (hospital)-[:EMPLOYS]->(physician)
    """)

# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    driver = get_driver()
    try:
        load_hospital_graph_from_csv(driver)
        LOGGER.info("Hospital graph loaded successfully")
    finally:
        driver.close()
