import asyncio
from config import get_settings
from parsers.langextract_pln_parser import LangExtractPLNParser
from core.langextract_pln import build_canonicalization_context, translate_query_extractions_to_pln

async def main():
    parser = LangExtractPLNParser()
    text = "Is Abebe consuming excessive carbohydrates?"
    print(parser.debug_parse_query(text, context=[]))

asyncio.run(main())
