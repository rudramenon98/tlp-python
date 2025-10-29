from pathlib import Path

import pandas as pd
from magic_draw_methods import (
    ClassPackage,
    MagicDrawProject,
    Package,
    RelationshipType,
)


def make_mdxml(df):   
    # Create a new MagicDraw project
    project = MagicDrawProject()
    
    architecture_blocks = {}
    
    # loop through the architecture blocks and add them to the project
    architecture_classes = []
    for _, row in df.iterrows():
        if row['type'] == 'Architecture Block':
            block_number = row['Requirement Number']
            mapping = row['Mapping to system architecture']
            name = f"{block_number} - {mapping}"
            body = row['Multiple requirements styled and separated']
            package = ClassPackage(name, body)
            block = project.add_block(package)
            architecture_blocks[row['Mapping to system architecture']] = package
            architecture_classes.append(package)
    print(architecture_blocks.keys())
    package = Package(architecture_classes, "Architecture Blocks")
    project.add_package(package)
        
    for arch, block in architecture_blocks.items():
        if ">" in arch:
            # get everything before the last '>'    
            arch = arch.rsplit('>', 1)[0].strip()
            if arch == "No mapping":
                continue
            project.add_relationship(architecture_blocks[arch], block, RelationshipType.COMPOSITION)
    # Process each row in the CSV
    requirements_classes = []
    for _, row in df.iterrows():    
        if row['type'] != 'Requirement':
            continue
        block_number = row['Requirement Number']
        key_phrases = row['Key Phrases Extracted']
        
        name = f"{block_number} - {key_phrases}"
        body = row['Multiple requirements styled and separated']
        if pd.notna(name) and pd.notna(body):
            package = ClassPackage(name, body)
            
            for key, value in row.items():
                if key != 'Requirement Number' and key != 'Key Phrases Extracted' and key != "Parsed text":
                    package.add_attribute(key, value)
            block = project.add_block(package)
            arch= row['Mapping to system architecture']
            if arch == "No mapping":
                continue
            print("Adding relationship to ", arch)
            parent_block = architecture_blocks[arch]
            project.add_relationship(package, parent_block, RelationshipType.ASSOCIATION)
            
            requirements_classes.append(package)
    requirements_package = Package(requirements_classes, "Requirements")
    project.add_package(requirements_package)
    
    return project
    

def main():
    output_dir = Path(__file__).parent / "web_app" / "output"
    csv_path = output_dir / "detailed_requirement_analysis.csv"
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    project = make_mdxml(df)
    
    # Save the project
    output_path = output_dir / "requirements.mdxml"
    print(f"Saving project to {output_path}")
    project.save(output_path)
    
if __name__ == "__main__":
    main()