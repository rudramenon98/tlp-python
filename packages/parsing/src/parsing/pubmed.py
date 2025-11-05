import os
import requests
import regex as re
import gzip
from lxml import etree
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Dict, Any

from datetime import datetime, date

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # You can add FileHandler here if needed
    ]
)

logger = logging.getLogger(__name__)

TODAY  = datetime.today().date()
@dataclass
class Author:
    last_name: Optional[str]
    fore_name: Optional[str]
    initials: Optional[str]
    affiliation: Optional[str]
    identifier: Optional[str] = None  # e.g. for AuthorIdentifier if present


@dataclass
class Grant:
    grant_id: Optional[str]
    acronym: Optional[str]
    agency: Optional[str]
    country: Optional[str]


@dataclass
class Chemical:
    name: Optional[str]
    registry_number: Optional[str]
    cas_registry_number: Optional[str]


@dataclass
class DataBank:
    db_name: Optional[str]
    accession_number: Optional[str]


@dataclass
class CommentOn:
    ref_pmid: Optional[str]
    ref_type: Optional[str]


@dataclass
class Reference:
    citation: Optional[str]
    ref_type: Optional[str]
    pubmed_id: Optional[str]


@dataclass
class Article:
    pmid: str
    deleted: Optional[bool]
    pmid_version: Optional[str]
    doi: Optional[str]
    title: Optional[str]
    abstract: Optional[str]
    other_abstracts: List[Dict[str, Any]]  # e.g. OtherAbstract elements (with language etc.)

    journal_title: Optional[str]
    journal_iso_abbrev: Optional[str]
    journal_pub_date: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    pagination: Optional[str]
    created_date: Optional[date]
    modified_date: Optional[date]

    # optional meta
    authors: List[Author] = field(default_factory=list)
    mesh_terms: List[Dict[str, Any]] = field(default_factory=list)  # with qualifiers / AutoHM etc.
    keywords: List[str] = field(default_factory=list)
    chemicals: List[Chemical] = field(default_factory=list)
    grants: List[Grant] = field(default_factory=list)
    databanks: List[DataBank] = field(default_factory=list)
    comments_on: List[CommentOn] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    citation_subset: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)



class PubmedFullParser:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path

    def parse(self) -> Iterator[Article]:
        """
        Iterate through <PubmedArticle> and <PubmedBookArticle> entries and yield Article objects.
        """
        tags = ("PubmedArticle", "PubmedBookArticle", "DeleteCitation")
        context = etree.iterparse(self.xml_path, events=("end",), tag=tags)
        for event, elem in context:
            try:
                if elem.tag == "DeleteCitation":
                    # These elements are expected to contain one or more PMID tags.
                    for child in elem.iterchildren():
                        assert child.tag == "PMID", f"PMID tag expected. Got: {child.tag}"
                        #yield {"pmid": child.text, "delete": True}
                        art = Article(
                            pmid=child.text,
                            deleted=True,
                            pmid_version='',
                            title='',
                            abstract='',
                            other_abstracts='',
                            journal_title='',
                            journal_iso_abbrev='',
                            journal_pub_date='',
                            volume='',
                            issue='',
                            pagination='',
                            created_date=TODAY,
                            modified_date=TODAY,
                            doi='',
                            )
                        yield art
                    elem.clear()
                else:
                    art = self._parse_pubmed_elem(elem)
                    elem.clear
                    if art is not None:
                        yield art
            except Exception as e:
                # handle or log errors
                print("Error parsing article:", e)
            # free memory as we go
            #elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context

    def _get_text(self, parent: etree._Element, path: str) -> Optional[str]:
        node = parent.find(path)
        if node is not None and node.text:
            return node.text.strip()
        return None

    def _parse_pubmed_elem(self, p: etree._Element) -> Optional[Article]:
        """
        Handle both PubmedArticle and PubmedBookArticle similarly (some fields may differ).
        """
        # Most data is under MedlineCitation
        med = p.find("MedlineCitation")
        if med is None:
            return None

        pmid_el = med.find("PMID")
        if pmid_el is None or pmid_el.text is None:
            return None
        pmid = pmid_el.text.strip()
        pmid_version = pmid_el.get("Version")

        def ParseDate(date_element):
            """Parse a **valid** date that (at least) has to have a Year element."""

            # translate three-letter month strings to integers:
            MONTHS_SHORT = (None, 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec')

            year = int(date_element.find('Year').text)
            month, day = 1, 1
            month_element = date_element.find('Month')
            day_element = date_element.find('Day')

            if month_element is not None:
                month_text = month_element.text.strip()
                try:
                    month = int(month_text)
                except ValueError:
                    logger.debug('non-numeric Month "%s"', month_text)
                    try:
                        month = MONTHS_SHORT.index(month_text.lower())
                    except ValueError:
                        logger.warning('could not parse Month "%s"', month_text)
                        month = 1

            if day_element is not None:
                try:
                    day = int(day_element.text)
                except (AttributeError, ValueError):
                    logger.warning('could not parse Day "%s"', day_element.text)

            return date(year, month, day)

        dates = {}
        for name, key in (('DateCompleted', 'completed'),
                          ('DateCreated', 'created'),
                          ('DateRevised', 'revised')):
            e = med.find(name)

            if e is not None:
                dates[key] = ParseDate(e)
        
        # Article (for PubmedArticle) or BookArticle (for PubmedBookArticle)
        article_node = med.find("Article")
        # Note: for BookArticle, you may have <BookDocument> or other structure — you can extend this.
        # In many Pubmed XMLs, BookArticle also uses "Article" sub‐structure.
        # We'll assume article_node is present for main bibliographic info.
        title = None
        abstract = None
        other_abstracts = []

        if article_node is not None:
            title = self._get_text(article_node, "ArticleTitle")

            # Abstract
            abstr = article_node.find("Abstract")
            if abstr is not None:
                parts = []
                for abst_el in abstr.findall("AbstractText"):
                    # AbstractText may have attributes (Label, NlmCategory, etc.)
                    txt = abst_el.text.strip() if abst_el.text else None
                    if txt:
                        parts.append(txt)
                if parts:
                    abstract = "\n".join(parts)
            # OtherAbstract (translated or alternate language abstracts)
            for oab in article_node.findall("OtherAbstract"):
                # attributes like Language etc.
                oa_lang = oab.get("Language")
                oa_text = "".join(oab.itertext()).strip()
                other_abstracts.append({"language": oa_lang, "text": oa_text})

            # get the DOI
            elocation_ids = article_node.findall("ELocationID")

            if len(elocation_ids) > 0:
                for e in elocation_ids:
                    doi = e.text.strip() or "" if e.attrib.get("EIdType", "") == "doi" else ""
            else:
                article_ids = med.find("PubmedData/ArticleIdList")
                if article_ids is not None:
                    doi = article_ids.find('ArticleId[@IdType="doi"]')
                    doi = (
                        (doi.text.strip() if doi.text is not None else "")
                        if doi is not None
                        else ""
                    )
                else:
                    doi = ""

        # Journal data
        journal = article_node.find("Journal") if article_node is not None else None
        journal_title = self._get_text(journal, "Title") if journal is not None else None
        journal_iso = self._get_text(journal, "ISOAbbreviation") if journal is not None else None
        journal_pub_date = None
        volume = None
        issue = None
        pagination = None

        if journal is not None:
            ji = journal.find("JournalIssue")
            if ji is not None:
                pd = ji.find("PubDate")
                if pd is not None:
                    year = self._get_text(pd, "Year")
                    month = self._get_text(pd, "Month")
                    day = self._get_text(pd, "Day")
                    if year:
                        temp_date = year
                        if month:
                            temp_date += "-" + month
                        if day:
                            temp_date += "-" + day
                        journal_pub_date = temp_date
                    else:
                        # fallback to MedlineDate
                        journal_pub_date = self._get_text(pd, "MedlineDate")
                volume = self._get_text(ji, "Volume")
                issue = self._get_text(ji, "Issue")
            pagination = self._get_text(article_node, "Pagination/MedlinePgn")

        # Authors
        authors: List[Author] = []
        if article_node is not None:
            author_list = article_node.find("AuthorList")
            if author_list is not None:
                for a in author_list.findall("Author"):
                    last = self._get_text(a, "LastName")
                    fore = self._get_text(a, "ForeName")
                    initials = self._get_text(a, "Initials")
                    # affiliation
                    affiliation = None
                    affinfo = a.find("AffiliationInfo")
                    if affinfo is not None:
                        affiliation = self._get_text(affinfo, "Affiliation")
                    # identifier (e.g. AuthorIdentifier)
                    ident = None
                    aid = a.find("AuthorIdentifier")
                    if aid is not None:
                        ident = aid.text.strip() if aid.text else None
                    authors.append(Author(last_name=last, fore_name=fore, initials=initials,
                                          affiliation=affiliation, identifier=ident))

        # Mesh headings & qualifiers
        mesh_terms = []
        mhlist = med.find("MeshHeadingList")
        if mhlist is not None:
            for mh in mhlist.findall("MeshHeading"):
                dname_el = mh.find("DescriptorName")
                if dname_el is None:
                    continue
                dname = dname_el.text.strip() if dname_el.text else None
                d_attrs = dict(dname_el.attrib)
                # qualifiers
                qualifiers = []
                for qn in mh.findall("QualifierName"):
                    q_text = qn.text.strip() if qn.text else None
                    q_attrs = dict(qn.attrib)
                    qualifiers.append({"qualifier_name": q_text, "attrs": q_attrs})
                mesh_terms.append({"descriptor": dname, "d_attrs": d_attrs, "qualifiers": qualifiers})

        # Keywords
        keywords: List[str] = []
        kwlist = med.find("KeywordList")
        if kwlist is not None:
            for kw in kwlist.findall("Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())

        # Chemicals
        chemicals: List[Chemical] = []
        chem_list = med.find("ChemicalList")
        if chem_list is not None:
            for chem in chem_list.findall("Chemical"):
                name = self._get_text(chem, "NameOfSubstance")
                registry = chem.find("RegistryNumber")
                cas = chem.find("RN")
                registry_number = registry.text.strip() if (registry is not None and registry.text) else None
                cas_number = cas.text.strip() if (cas is not None and cas.text) else None
                chemicals.append(Chemical(name=name, registry_number=registry_number,
                                          cas_registry_number=cas_number))

        # Grants
        grants: List[Grant] = []
        grant_list = med.find("ChemicalList")  # mistake — grants are under Article => GrantList under Article or under MedlineCitation?
        # Actually, in the DTD, GrantList is under MedlineCitation / Article (or under Article in some versions)
        # Better to search:
        for gl in med.findall(".//GrantList"):
            for grant in gl.findall("Grant"):
                gid = self._get_text(grant, "GrantID")
                acronym = grant.get("Acronym")
                agency = self._get_text(grant, "Agency")
                country = self._get_text(grant, "Country")
                grants.append(Grant(grant_id=gid, acronym=acronym, agency=agency, country=country))

        # DataBanks
        databanks: List[DataBank] = []
        for dblist in med.findall(".//DataBankList"):
            for db in dblist.findall("DataBank"):
                name = self._get_text(db, "DataBankName")
                for acc in db.findall("AccessionNumber"):
                    ani = acc.text.strip() if acc.text else None
                    databanks.append(DataBank(db_name=name, accession_number=ani))

        # CommentsOn / CommentList
        comments_on: List[CommentOn] = []
        cmnts = med.find("CommentsCorrectionsList")
        if cmnts is not None:
            for cc in cmnts.findall("CommentsCorrections"):
                ref_pmid = self._get_text(cc, "PMID")
                ref_type = cc.get("RefType")
                comments_on.append(CommentOn(ref_pmid=ref_pmid, ref_type=ref_type))

        # References — some citations list in <ReferenceList> under MedlineCitation
        references: List[Reference] = []
        reflist = med.find("ReferenceList")
        if reflist is not None:
            for ref in reflist.findall("Reference"):
                citation = self._get_text(ref, "Citation")
                rtype = ref.get("Type")
                pid = self._get_text(ref, "ArticleIdList/ArticleId[@IdType='pubmed']")
                references.append(Reference(citation=citation, ref_type=rtype, pubmed_id=pid))

        # CitationSubset — (e.g. “IM”, “OldMedline”, etc.)
        citation_subset = []
        for cs in med.findall("CitationSubset"):
            if cs.text:
                citation_subset.append(cs.text.strip())

        # Publication types (under Article / PublicationTypeList)
        publication_types: List[str] = []
        if article_node is not None:
            ptl = article_node.find("PublicationTypeList")
            if ptl is not None:
                for pt in ptl.findall("PublicationType"):
                    if pt.text:
                        publication_types.append(pt.text.strip())

        # Build Article instance
        art = Article(
            pmid=pmid,
            deleted=False,
            pmid_version=pmid_version,
            title=title,
            abstract=abstract,
            other_abstracts=other_abstracts,
            journal_title=journal_title,
            journal_iso_abbrev=journal_iso,
            journal_pub_date=journal_pub_date,
            volume=volume,
            issue=issue,
            pagination=pagination,
            created_date=dates.get('created', TODAY),
            modified_date=dates.get('revised', TODAY),

            authors=authors,
            mesh_terms=mesh_terms,
            keywords=keywords,
            chemicals=chemicals,
            grants=grants,
            databanks=databanks,
            comments_on=comments_on,
            references=references,
            citation_subset=citation_subset,
            publication_types=publication_types,
            doi=doi,
        )

        return art

def process_pubmed_update_file(update_file_path):
    """
    Process a PubMed XML update file, logging information about each article.
    
    Parameters:
        update_file_path (str): Full path to the .xml.gz update file.
    """
    # Validate the input path
    if not os.path.isfile(update_file_path):
        logger.error(f"File not found: {update_file_path}")
        return

    if not update_file_path.endswith('.xml.gz'):
        logger.error("Invalid file format. Expected a .xml.gz compressed file.")
        return

    try:
        with gzip.open(update_file_path, "rb") as f:
            parser = PubmedFullParser(f)
            for art in parser.parse():
                try:
                    if art.deleted:
                        logger.info(f"[DELETED] PMID {art.pmid}, Title: {art.title}")
                    
                    elif art.abstract and art.doi:
                        if int(art.pmid_version) > 1:
                            logger.info(f"[UPDATED VERSION] PMID {art.pmid}, Version: {art.pmid_version}")
                        else:
                            logger.info(f"[NEW ENTRY] PMID {art.pmid}, Title: {art.title}")
                            logger.info(f"Created Date: {art.created_date}")
                            logger.info(f"Revised Date: {art.modified_date}")
                            logger.info("-----")

                except Exception as article_error:
                    logger.warning(f"Error processing article: {article_error}")
                    continue

    except Exception as e:
        logger.exception(f"Failed to process file '{update_file_path}': {e}")

# Example usage:
if __name__ == "__main__":
    
    file_path = '/uploads/Enginius/test/scrapedDocs/pubmed/baseline_files/pubmed25n0001.xml.gz'
    update_file_path = '/uploads/Enginius/test/scrapedDocs/pubmed/update_files/pubmed25n1278.xml.gz'
    with gzip.open(update_file_path, "rb") as f:
        parser = PubmedFullParser(f)
        for art in parser.parse():
            if art.deleted:
                print(f"PMID {art.pmid}, Title: {art.title}")
                print(f"Deleted: {art.deleted}")

            elif art.abstract and len(art.abstract) > 0 and art.doi and len(art.doi) > 0:
                if int(art.pmid_version) > 1:
                    print(f"PMID {art.pmid}, PMID Version: {art.pmid_version}")
                else:
                    pass
                    print(f"PMID {art.pmid}, Title: {art.title}")
                    #print(f"Abstract: {art.abstract}")
                    #print(f"DOI: https://doi.org/{art.doi}")
                    print(f"Created Date: {art.created_date}")
                    print(f"Revised Date: {art.modified_date}")
                    '''
                    print("Authors:", [(a.last_name, a.fore_name) for a in art.authors])
                    print("Mesh:", [m["descriptor"] for m in art.mesh_terms])
                    print("Chemicals:", [(c.name, c.cas_registry_number) for c in art.chemicals])
                    print("Grants:", [(g.grant_id, g.agency) for g in art.grants])
                    print("DataBanks:", [(d.db_name, d.accession_number) for d in art.databanks])
                    print("References:", [(r.pubmed_id, r.citation) for r in art.references])
                    '''
                    print("-----")
