
import langextract as lx



EXTRACTION_PROMPT = """\
Extract semantic facts, type declarations, inheritance relationships, rules, and negations from the text.

Extraction classes:
- 'fact': a predicate with one or more arguments
  - Binary example: "Sam is a frog" -> subject=Sam, predicate=isa, object=frog
  - Unary example: "Sam croaks" -> subject=Sam, predicate=croaks (omit object)
  
- 'type_decl': declares the MeTTa type of an entity using ':'
  e.g. "Sam is of type Animal" -> entity=Sam, type=Animal
  
- 'property': assigns a property / attribute value to an entity
  e.g. "Tom is white" -> entity=Tom, property=color, value=white
  
- 'inheritance': subtype or kind-of relationship
  e.g. "frog is a kind of animal" -> child=frog, parent=animal
- 'rule': a conditional implication
  e.g. "every frog that croaks and eats flies is green"
       -> head_predicate=has, head_args="$x color green",
          body=(and (isa $x frog) (croaks $x) (eats-flies $x))
- 'negation': explicit negation of a unary or binary fact
  - Binary example: "Tom is not a frog" -> subject=Tom, predicate=isa, object=frog
  - Unary example: "Sam cannot fly" -> subject=Sam, predicate=can-fly

Rules for attribute values:
- Use lowercase, hyphen-separated names for predicates and property names
  (e.g. eats-flies, has-wings, can-fly, student-of, part-of)
- Entity and type names may preserve capitalization, but replace internal spaces with hyphens
- Hyphens and digits inside source identifiers are preserved verbatim
  (e.g. "Object-77", "rack-01", "vault-07")
- Variables in rules must start with $ (e.g. $x, $who, $topic)
- Rule head arguments must be space-separated in head_args
- Rule body must be a valid MeTTa S-expression; use (and ...) for conjunctions
- Multi-premise rules share variables across premises to express joins
  (e.g. body=(and (student-of $s $t) (expert-in $t $topic)) -> head shares $s and $topic)
- Definitional sentences ("X means Y", "X is when Y") become 'rule' with a single-atom body
  (e.g. "Freezing means a substance is decreasing in heat" -> head=decreasing-heat, body=(is-freezing $s))
- For transitive chains ("A causes B causes C"), emit each link as a separate 'rule';
  do not collapse multiple steps into one
- Use exact source text spans for extraction_text; do not paraphrase
"""


METTA_EXAMPLES = [

    # Facts: binary relations and unary predicates.
    lx.data.ExampleData(
        text="Sam is a frog. Tom is a cat. Bob is a dog. Sam croaks. Bob eats flies.",
        extractions=[
            lx.data.Extraction(
                "fact", "Sam is a frog",
                attributes={"subject": "Sam", "predicate": "isa", "object": "frog"},
            ),
            lx.data.Extraction(
                "fact", "Tom is a cat",
                attributes={"subject": "Tom", "predicate": "isa", "object": "cat"},
            ),
            lx.data.Extraction(
                "fact", "Bob is a dog",
                attributes={"subject": "Bob", "predicate": "isa", "object": "dog"},
            ),
            lx.data.Extraction(
                "fact", "Sam croaks",
                attributes={"subject": "Sam", "predicate": "croaks"},
            ),
            lx.data.Extraction(
                "fact", "Bob eats flies",
                attributes={"subject": "Bob", "predicate": "eats-flies"},
            ),
        ],
    ),

    # Type declarations and properties, including adjectival colors.
    lx.data.ExampleData(
        text=(
            "Sam is of type Animal. Alice has type Person. "
            "Tom is white. Sam has age 3."
        ),
        extractions=[
            lx.data.Extraction(
                "type_decl", "Sam is of type Animal",
                attributes={"entity": "Sam", "type": "Animal"},
            ),
            lx.data.Extraction(
                "type_decl", "Alice has type Person",
                attributes={"entity": "Alice", "type": "Person"},
            ),
            lx.data.Extraction(
                "property", "Tom is white",
                attributes={"entity": "Tom", "property": "color", "value": "white"},
            ),
            lx.data.Extraction(
                "property", "Sam has age 3",
                attributes={"entity": "Sam", "property": "age", "value": "3"},
            ),
        ],
    ),

    # Inheritance / taxonomy.
    lx.data.ExampleData(
        text=(
            "A frog is a kind of animal. Cats are mammals. "
            "Dogs are a subtype of mammal. Every mammal is an animal."
        ),
        extractions=[
            lx.data.Extraction(
                "inheritance", "A frog is a kind of animal",
                attributes={"child": "frog", "parent": "animal"},
            ),
            lx.data.Extraction(
                "inheritance", "Cats are mammals",
                attributes={"child": "cat", "parent": "mammal"},
            ),
            lx.data.Extraction(
                "inheritance", "Dogs are a subtype of mammal",
                attributes={"child": "dog", "parent": "mammal"},
            ),
            lx.data.Extraction(
                "inheritance", "Every mammal is an animal",
                attributes={"child": "mammal", "parent": "animal"},
            ),
        ],
    ),

    # Conditional rules. The body is expressed as valid MeTTa S-expression.
    lx.data.ExampleData(
        text=(
            "Every frog that croaks and eats flies is green. "
            "If an animal has wings it can fly."
        ),
        extractions=[
            lx.data.Extraction(
                "rule",
                "Every frog that croaks and eats flies is green",
                attributes={
                    "head_predicate": "has",
                    "head_args": "$x color green",
                    "body": "(and (isa $x frog) (croaks $x) (eats-flies $x))",
                },
            ),
            lx.data.Extraction(
                "rule",
                "If an animal has wings it can fly",
                attributes={
                    "head_predicate": "can-fly",
                    "head_args": "$x",
                    "body": "(and (isa $x animal) (has-wings $x))",
                },
            ),
        ],
    ),

    # Negation of binary and unary predicates.
    lx.data.ExampleData(
        text="Tom is not a frog. Sam cannot fly.",
        extractions=[
            lx.data.Extraction(
                "negation", "Tom is not a frog",
                attributes={"subject": "Tom", "predicate": "isa", "object": "frog"},
            ),
            lx.data.Extraction(
                "negation", "Sam cannot fly",
                attributes={"subject": "Sam", "predicate": "can-fly"},
            ),
        ],
    ),

    # Hyphenated/numeric proper nouns + transitive rule chain.
    lx.data.ExampleData(
        text=(
            "Object-77 is fragile. Fragile items need a double wrap. "
            "Items that need a double wrap are in a padded box."
        ),
        extractions=[
            lx.data.Extraction(
                "fact", "Object-77 is fragile",
                attributes={"subject": "Object-77", "predicate": "fragile"},
            ),
            lx.data.Extraction(
                "rule",
                "Fragile items need a double wrap",
                attributes={
                    "head_predicate": "needs-double-wrap",
                    "head_args": "$x",
                    "body": "(fragile $x)",
                },
            ),
            lx.data.Extraction(
                "rule",
                "Items that need a double wrap are in a padded box",
                attributes={
                    "head_predicate": "in-padded-box",
                    "head_args": "$x",
                    "body": "(needs-double-wrap $x)",
                },
            ),
        ],
    ),

    # Multi-argument binary relations + 2-premise conjunction rule sharing variables.
    lx.data.ExampleData(
        text=(
            "Kebede is a student of Ayele. Ayele is an expert in quantum physics. "
            "A student of an expert knows the topic."
        ),
        extractions=[
            lx.data.Extraction(
                "fact", "Kebede is a student of Ayele",
                attributes={"subject": "Kebede", "predicate": "student-of", "object": "Ayele"},
            ),
            lx.data.Extraction(
                "fact", "Ayele is an expert in quantum physics",
                attributes={"subject": "Ayele", "predicate": "expert-in", "object": "quantum physics"},
            ),
            lx.data.Extraction(
                "rule",
                "A student of an expert knows the topic",
                attributes={
                    "head_predicate": "knows",
                    "head_args": "$s $topic",
                    "body": "(and (student-of $s $t) (expert-in $t $topic))",
                },
            ),
        ],
    ),

    # Spatial location chain with hyphenated/numeric identifiers.
    lx.data.ExampleData(
        text=(
            "Kebede is at Rack-01. Rack-01 is inside Vault-07. "
            "A person at a spot inside an area is in that area."
        ),
        extractions=[
            lx.data.Extraction(
                "fact", "Kebede is at Rack-01",
                attributes={"subject": "Kebede", "predicate": "at-spot", "object": "Rack-01"},
            ),
            lx.data.Extraction(
                "fact", "Rack-01 is inside Vault-07",
                attributes={"subject": "Rack-01", "predicate": "inside", "object": "Vault-07"},
            ),
            lx.data.Extraction(
                "rule",
                "A person at a spot inside an area is in that area",
                attributes={
                    "head_predicate": "in-area",
                    "head_args": "$p $a",
                    "body": "(and (at-spot $p $s) (inside $s $a))",
                },
            ),
        ],
    ),

    # Part-of fact + inheritance + multi-premise rule combining them.
    lx.data.ExampleData(
        text=(
            "An electric motor is a part of an electric fan. "
            "An electric fan is a kind of appliance. "
            "If a part converts energy and is part of a whole, the whole converts energy."
        ),
        extractions=[
            lx.data.Extraction(
                "fact", "An electric motor is a part of an electric fan",
                attributes={
                    "subject": "electric motor",
                    "predicate": "part-of",
                    "object": "electric fan",
                },
            ),
            lx.data.Extraction(
                "inheritance", "An electric fan is a kind of appliance",
                attributes={"child": "electric fan", "parent": "appliance"},
            ),
            lx.data.Extraction(
                "rule",
                "If a part converts energy and is part of a whole, the whole converts energy",
                attributes={
                    "head_predicate": "converts-energy",
                    "head_args": "$whole",
                    "body": "(and (part-of $part $whole) (converts-energy $part))",
                },
            ),
        ],
    ),

    # Definitional rule + multi-step transitive chain.
    lx.data.ExampleData(
        text=(
            "Freezing means a substance is decreasing in heat. "
            "As heat decreases, temperature decreases. "
            "As temperature decreases, molecules move slower. "
            "Water is freezing."
        ),
        extractions=[
            lx.data.Extraction(
                "rule",
                "Freezing means a substance is decreasing in heat",
                attributes={
                    "head_predicate": "decreasing-heat",
                    "head_args": "$s",
                    "body": "(is-freezing $s)",
                },
            ),
            lx.data.Extraction(
                "rule",
                "As heat decreases, temperature decreases",
                attributes={
                    "head_predicate": "decreasing-temp",
                    "head_args": "$s",
                    "body": "(decreasing-heat $s)",
                },
            ),
            lx.data.Extraction(
                "rule",
                "As temperature decreases, molecules move slower",
                attributes={
                    "head_predicate": "molecules-move-slower",
                    "head_args": "$s",
                    "body": "(decreasing-temp $s)",
                },
            ),
            lx.data.Extraction(
                "fact", "Water is freezing",
                attributes={"subject": "Water", "predicate": "is-freezing"},
            ),
        ],
    ),
]

