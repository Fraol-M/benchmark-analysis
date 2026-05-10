import asyncio
from config import get_settings
from core.service import PLNRAGService
from parsers.langextract_pln_parser import LangExtractPLNParser

async def main():
    parser = LangExtractPLNParser()
    service = PLNRAGService(parser)
    text = "People who frequently eat pasta tend to consume higher amounts of carbohydrates, especially if the portions are large and the pasta is refined rather than whole grain."
    print("Ingesting...")
    res = await service.debug_ingest_batch([text])
    print("Rule added:")
    for stmt in res[0].chunks[0].atomspace_added:
        print(stmt)
asyncio.run(main())
