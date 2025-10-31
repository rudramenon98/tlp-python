import os
import sys

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import traceback

from database.entity.Dictionary import Base, Definition


def Error_Handler(fn):
    def Inner_Function(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
            return ret
        except Exception:
            print(f"Exception in {fn.__name__}")
            traceback.print_exc()

    return Inner_Function


@Error_Handler
def find_definition_by_id(mysql_driver, definition_id):
    session = mysql_driver.get_session()
    result = session.query(Definition).filter(Definition.id == definition_id).first()
    session.close()
    return result


@Error_Handler
def find_definition_by_term(mysql_driver, term):
    session = mysql_driver.get_session()
    result = session.query(Definition).filter(Definition.term == term).first()
    session.close()
    return result


@Error_Handler
def find_definition_by_term_and_type(mysql_driver, term, type):
    session = mysql_driver.get_session()
    result = (
        session.query(Definition)
        .filter(Definition.term == term)
        .filter(Definition.type == type)
        .first()
    )
    session.close()
    return result


@Error_Handler
def find_definitions_by_category(mysql_driver, category_id):
    session = mysql_driver.get_session()
    result = (
        session.query(Definition).filter(Definition.categoryID == category_id).all()
    )
    session.close()
    return result


@Error_Handler
def find_expansions_by_source_id(mysql_driver, source_id, type):
    session = mysql_driver.get_session()
    result = (
        session.query(Definition)
        .filter(Definition.sourceId == source_id)
        .filter(Definition.type == type)
        .all()
    )
    session.close()
    return result


@Error_Handler
def find_definitions_by_type(mysql_driver, definition_type):
    session = mysql_driver.get_session()
    result = session.query(Definition).filter(Definition.type == definition_type).all()
    session.close()
    return result


@Error_Handler
def find_approved_definitions(mysql_driver):
    session = mysql_driver.get_session()
    result = session.query(Definition).filter(Definition.approved == True).all()
    session.close()
    return result


@Error_Handler
def insert_definition(mysql_driver, definition):
    session = mysql_driver.get_session()
    session.add(definition)
    session.commit()
    session.close()


@Error_Handler
def insert_definitions_bulk(mysql_driver, definition_list):
    session = mysql_driver.get_session()
    for definition in definition_list:
        session.add(definition)
        session.commit()
    session.close()


@Error_Handler
def insert_definitions_bulk2(mysql_driver, definition_list):
    definition_id_list = []
    session = mysql_driver.get_session()
    for definition in definition_list:
        if definition is None:
            continue
        print(f"Inserting definition into DB {definition.term}")
        session.add(definition)
        session.commit()
        definition_id_list.append(definition.id)
    session.close()
    return definition_id_list


@Error_Handler
def get_all_definitions(mysql_driver):
    session = mysql_driver.get_session()
    result = session.query(Definition).all()
    session.close()
    return result


@Error_Handler
def update_definition(mysql_driver, definition):
    session = mysql_driver.get_session()
    session.query(Definition).filter(Definition.id == definition.id).update(
        {
            "term": definition.term,
            "categoryID": definition.categoryID,
            "type": definition.type,
            "sourceId": definition.sourceId,
            "expansionCount": definition.expansionCount,
            "repoFreq": definition.repoFreq,
            "uploadDocName": definition.uploadDocName,
            "tdVocabId": definition.tdVocabId,
            "tsseVocabId": definition.tsseVocabId,
            "tdVocabAddDate": definition.tdVocabAddDate,
            "modifiedDate": definition.modifiedDate,
            #            'approved': definition.approved
        }
    )
    session.commit()
    session.close()


@Error_Handler
def set_definition_as_approved(mysql_driver, definition_id):
    session = mysql_driver.get_session()
    session.query(Definition).filter(Definition.id == definition_id).update(
        {"approved": True}
    )
    session.commit()
    session.close()


@Error_Handler
def delete_definition(mysql_driver, definition_id):
    session = mysql_driver.get_session()
    session.query(Definition).filter(Definition.id == definition_id).delete()
    session.commit()
    session.close()


@Error_Handler
def delete_all_definitions(mysql_driver):
    session = mysql_driver.get_session()
    session.query(Definition).delete()
    session.commit()
    session.close()


@Error_Handler
def create_definitions_table(mysql_driver):
    """Create the definitions table if it doesn't exist"""
    try:
        # Get the engine from the session
        session = mysql_driver.get_session()
        engine = session.bind

        # Create the table
        Base.metadata.create_all(engine, tables=[Definition.__table__])
        print("Definitions table created successfully")
        session.close()
        return True
    except Exception as e:
        print(f"Error creating definitions table: {e}")
        session.close()
        return False


@Error_Handler
def get_definitions_by_source(mysql_driver, source_id):
    session = mysql_driver.get_session()
    result = session.query(Definition).filter(Definition.sourceId == source_id).all()
    session.close()
    return result


@Error_Handler
def get_definitions_by_td_vocab_id(mysql_driver, td_vocab_id):
    session = mysql_driver.get_session()
    result = session.query(Definition).filter(Definition.tdVocabId == td_vocab_id).all()
    session.close()
    return result


@Error_Handler
def get_definitions_by_tsse_vocab_id(mysql_driver, tsse_vocab_id):
    session = mysql_driver.get_session()
    result = (
        session.query(Definition).filter(Definition.tsseVocabId == tsse_vocab_id).all()
    )
    session.close()
    return result


# Driver code example - uncomment and modify as needed
if __name__ == "__main__":
    from datetime import datetime

    from database.entity.ScriptsProperty import parseCredentialFile
    from database.utils.MySQLFactory import MySQLDriver

    # Load configuration
    config = parseCredentialFile("scripts/app/tlp_config.json")

    if config is None:
        print("Error: Could not load configuration file")
        exit(1)

    # Initialize database driver
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    try:
        # Create the definitions table if it doesn't exist
        print("Creating definitions table...")
        create_definitions_table(mysql_driver)

        # Create an example definition
        example_definition = Definition(
            term="USA",
            categoryID=1,
            type=1,
            sourceId=8,
            expansionCount=0,
            repoFreq=1,
            uploadDocName="",
            tdVocabId=None,
            tsseVocabId=None,
            tdVocabAddDate=None,
            createdDate=datetime.now(),
            modifiedDate=datetime.now(),
            #            approved=False
        )

        # Insert the definition into the database
        print("Inserting example definition...")
        print(f"Successfully inserted definition: {example_definition.term}")
        insert_definition(mysql_driver, example_definition)

        # Retrieve and display the inserted definition
        retrieved_definition = find_definition_by_term(mysql_driver, "Medical Device")
        if retrieved_definition:
            print(
                f"Retrieved definition - ID: {retrieved_definition.id}, Term: {retrieved_definition.term}"
            )
            print(
                f"Category: {retrieved_definition.categoryID}, Type: {retrieved_definition.type}"
            )
            print(f"Approved: {retrieved_definition.approved}")
        else:
            print("Definition not found after insertion")

        # Print all definitions in the database
        print("\n--- All Definitions in Database ---")
        all_definitions = get_all_definitions(mysql_driver)
        if all_definitions:
            for i, definition in enumerate(all_definitions, 1):
                print(
                    f"{i}. ID: {definition.id}, Term: '{definition.term}', Type: {definition.type}, Approved: {definition.approved}"
                )
        else:
            print("No definitions found in database")

    except Exception as e:
        print(f"Error in driver code: {e}")
        traceback.print_exc()
    finally:
        # Close database connection
        mysql_driver.close()
        print("Database connection closed")
