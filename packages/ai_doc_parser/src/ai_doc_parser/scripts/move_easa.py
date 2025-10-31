import shutil
from pathlib import Path

xml_fnames = [
    "03c2c2_2023-09-27_09.xml",
    "0934BC_2025-03-12_12.xml",
    "313A4D_2024-07-01_21.xml",
    "4E4220_2025-02-13_18.xml",
    "51757e_2023-10-27_10.xml",
    "532F14_2024-12-02_11.xml",
    "7AD833_2024-12-12_11.xml",
    "95CADA_2024-07-17_12.xml",
    "9ed0bb_2024-10-30_12.xml",
    "CEB6F1_2024-12-08_23.xml",
    "cs-fcd_june_2023_xml_final.xml",
    "D0593E_2024-07-10_06.xml",
    "EAR for CS-25 Amdt 27 (xml) fix 12.xml",
    "Easy Access Rules for the Basic Regulation (Regulation (EU) 2018-1139) - xml (machine readable).xml",
    "easy_access_rules_for_acceptable_means_of_compliance_for_airworthiness_of_products_parts_and_appliances_amc-20_amendment_23_xml.xml",
    "easy_access_rules_for_airborne_communications_navigation_and_surveillance_cs-acns.xml",
    "Easy_Access_Rules_for_Air_Traffic_Controllers__Licensing_and_Certification_-_revision_March_2024_-_correction_June_2024__xml_.xml",
    "easy_access_rules_for_balloons.xml",
    "easy_access_rules_for_cs-23_amendment_6_amc-gm_issue_4_xml.xml",
    "easy_access_rules_for_cs-27_amdt_10_xml.xml",
    "easy_access_rules_for_cs-29_amendment_11_-_december_2023_xml.xml",
    "easy_access_rules_for_cs-gen-mmel_issue_2_xml.xml",
    "easy_access_rules_for_cs-mmel_issue_3_xml.xml",
    "Easy_Access_Rules_for_Information_Security_download.xml",
    "Easy_Access_Rules_for_Part-26_Revision_March_23__xml__0.xml",
    "Easy_Access_Rules_for_Third_Country_Operators_-_Revision_from_April_2023__XML__0.xml",
    "Easy_Access_Rules_for_U-space_-_May_2024__XML_.xml",
    "FC9A84_2024-11-25_11.xml",
]

pdf_fnames = [
    "Easy Access Rules for Occurrence Reporting _Regulation _EU_ No 376_2014_ _ Revision from December 2022 _PDF_.pdf",
    "Easy Access Rules for ATM_ANS _ Provision of Services _Regulation _EU_ 2017_373_ _PDF_.pdf",
    "Easy Access Rules for Initial Airworthiness and Environmental Protection _Regulation _EU_ No 748_2012_ _PDF_.pdf",
    "Easy Access Rules for Air Operations _PDF_.pdf",
    "Easy Access Rules for Hot Air Balloons _CS_31HB_ _Amendment 1_ _pdf_.pdf",
    "Easy Access Rules for Standardised European Rules of the Air _SERA_ _PDF_.pdf",
    "Easy Access Rules for Aerodromes _PDF_.pdf",
    "Easy Access Rules for Continuing Airworthiness _PDF_.pdf",
    "Easy Access Rules for small category VCA _PDF_.pdf",
    "Easy Access Rules for Aircrew _Regulation _EU_ No 1178_2011_ _PDF_.pdf",
    "Easy Access Rules for Operational Suitability Data _OSD_ Flight Crew Data _CS_FCD_ _Issue 2_ _pdf_.pdf",
    "Easy Access Rules for Unmanned Aircraft Systems _Regulation _EU_ 2019_947 and Regulation _EU_ 2019_945_ _PDF_.pdf",
    "Easy Access Rules for Large Aeroplanes _CS 25_ _Amendment 27_ _PDF_.pdf",
    "Easy Access Rules for the Basic Regulation _Regulation _EU_ 2018_1139_ _PDF_.pdf",
    "Easy Access Rules for Acceptable Means of Compliance for Airworthiness of Products_ Parts and Appliances _AMC_20_ Amdt 23 _PDF_.pdf",
    "Easy Access Rules for Airborne Communications_ Navigation and Surveillance _CS_ACNS_ Issue 4 _pdf_.pdf",
    "Easy Access Rules for Air Traffic Controllers_ Licensing and Certification _Regulation _EU_ 2015_340_ _PDF_.pdf",
    "Balloon Rule Book _ Easy Access Rules _PDF_.pdf",
    "Easy Access Rules for Normal_Category Aeroplanes _CS_23_ _CS Amendment 6_ AMC_GM Issue 4_ _PDF_.pdf",
    "Easy Access Rules for Small Rotorcraft _CS_27_ Amendment 10 _pdf_.pdf",
    "Easy Access Rules for Large Rotorcraft _CS_29_ _Amendment 11_ _PDF_.pdf",
    "Easy Access Rules for Generic Master Minimum Equipment List _CS_GEN_MMEL_ _Issue 2_ _PDF_.pdf",
    "Easy Access Rules for Master Minimum Equipment List _CS_MMEL_ _Issue 3_ _PDF_.pdf",
    "Easy Access Rules for Information Security _PDF_.pdf",
    "Easy Access Rules for Additional Airworthiness Specifications _Regulation _EU_ 2015_640_ _ Revision from March 2023 _PDF_.pdf",
    "Easy Access Rules for Third Country Operators _ Revision from April 2023 _PDF_.pdf",
    "Easy Access Rules for U_space _PDF_.pdf",
    "Easy Access Rules for ATM_ANS Equipment _Regulations _EU_ 2023_1769 _ _EU_ 2023_1768_ _PDF_.pdf",
]

path = "/uploads/Enginius/test/scrapedDocs/"
data_path = "/home/rmenon/source/ai-pdf-parser/data"

for pdf_fname, xml_fname in zip(pdf_fnames, xml_fnames):
    # if both names are in path, copy them to the data folder
    pdf_path = Path(path) / pdf_fname
    xml_path = Path(path) / xml_fname
    new_pdf_path = Path(data_path) / pdf_fname
    new_xml_path = Path(data_path) / f"{pdf_path.stem}.xml"
    if pdf_path.exists() and xml_path.exists():
        shutil.copy(pdf_path, new_pdf_path)
        shutil.copy(xml_path, new_xml_path)