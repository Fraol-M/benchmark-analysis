import asyncio
from config import get_settings
from parsers.langextract_pln_parser import LangExtractPLNParser

async def main():
    parser = LangExtractPLNParser()
    text = "Abebe eats pasta almost every day, often in large portions, and prefers refined pasta over whole grain."
    print("Parsing text:")
    res = parser.debug_parse(text, [])
    for stmt in res['langextract_postprocessed']['statements']:
        print(stmt)

asyncio.run(main())
