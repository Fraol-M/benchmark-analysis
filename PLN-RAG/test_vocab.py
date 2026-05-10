import asyncio
from parsers.langextract_pln_parser import LangExtractPLNParser

async def main():
    parser = LangExtractPLNParser()
    text = "People who frequently eat pasta tend to consume higher amounts of carbohydrates, especially if the portions are large and the pasta is refined rather than whole grain."
    print("Rule:")
    res = parser.debug_parse(text, [])
    for stmt in res['langextract_postprocessed']['statements']:
        print(stmt)
        
    print("\nQuery:")
    query_text = "Is Abebe consuming excessive carbohydrates?"
    res_q = parser.debug_parse_query(query_text, context=["(: x_consume_higher_amount_of_carbohydrate_rule (Implication (Premises (EatsFrequently $x $p) (IsA $p pasta) (EatsLargePortions $x) (IsRefined $p) (Not (IsWholeGrain $p))) (Conclusions (ConsumeHigherAmountOfCarbohydrate $x))) (STV 1.0 1.0))"])
    for stmt in res_q['pln_canonicalized']:
        print(stmt)

asyncio.run(main())
