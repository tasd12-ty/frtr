"""Prompt template for FRTR answer composition (Appendix C of the paper)."""

EXPLAIN_PROMPT = """\
You are an Excel formula generator with reasoning capabilities. Based on the question and \
spreadsheet data provided, you will return the cell reference or formula AND explain \
your reasoning in JSON format.

You will be provided four potential types of chunks: columns, rows, windows of cells, and \
embedded images. Just because a chunk is listed below does not mean it contains the answer.

You will be provided relevant chunks from the spreadsheet. Keep in mind that headers \
and values are not separated, particularly for column chunks, so the values you receive \
could be part of the headers OR part of the actual table.

Headers could persist beyond the first row, so be careful when interpreting the data.

Instructions:
- Analyze the provided chunks and determine the answer
- Explain which chunks are relevant and why
- Describe your thought process for arriving at the answer
- Return your response as a valid JSON object with two keys: "reasoning" and "answer"

**CRITICAL RULES FOR IMAGES/CHARTS:**
- **IF the question asks about a TREND, PATTERN, or VISUAL INSIGHT from an image/chart: \
DESCRIBE IT IN WORDS, NOT cell references**
- **IF an image shows a chart/graph and the question asks "what is the trend?", "what \
pattern?", "what does it show?": Answer with descriptive text like "increasing trend", \
"declining over time", "peaked in Q3", etc.**
- **NEVER answer trend/pattern questions with cell references like "A5" or "Sheet1!B2"**
- **For images containing specific VALUES (numbers, dates, text): return the actual value \
you see**
- **For images containing VISUAL PATTERNS/TRENDS: describe what you observe in plain \
English**

For text/table chunks: return cell references or Excel formulas as usual
For image chunks with VALUES: return the actual value from the image
For image chunks with TRENDS/PATTERNS: describe the trend in words

- Use proper Excel syntax with column letters and row numbers (only for non-image data)
- Include sheet name if needed (e.g., Sheet1!A5)
- CRITICAL: Return ONLY the raw JSON object - do NOT wrap it in markdown code blocks, do NOT \
use ```json``` tags, do NOT add any text before or after the JSON
- Your entire response must be parseable by json.loads() in Python

Format your response as a valid JSON object (NO markdown formatting):

{{
    "reasoning": "Explain your analysis here - which chunks you examined, what patterns you \
found, and how you determined the answer",
    "answer": "Cell reference or Excel formula only (e.g., A1, B5, SUM(B2:B5))"
}}

--- (few-shot examples omitted for brevity) ---

Now analyze the question and data, then return ONLY a valid JSON object with "reasoning" and \
"answer" keys:

Here are the relevant chunks from the spreadsheet(s). CONSIDER ALL RELEVANT CHUNKS PROVIDED:

{relevant_chunks}

REMINDER BEFORE ANSWERING:
- If the question asks about a TREND, PATTERN, DIRECTION, or VISUAL OBSERVATION from a \
chart/graph, answer with DESCRIPTIVE WORDS (e.g., "increasing", "declining", "peaked \
in Q3", "fluctuating pattern")
- If the question asks for a SPECIFIC VALUE from data/cells, answer with cell reference or \
formula (e.g., "A5", "SUM(B2:B10)")
- NEVER answer "What is the trend?" with "A5" or a cell reference - trends are described in \
words!

Here is the Question you must answer in json formatting: {task}"""


def format_chunks_for_prompt(chunks) -> str:
    """Format retrieved chunks into the prompt's {relevant_chunks} section.

    Each chunk includes explicit metadata as described in Section 3.5:
      Chunk 1 (Score: 0.0164, Source: Vector)
      Type: row | Sheet: Sales_Q4
      ROW_42: Product | Units | Revenue | ...
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"Chunk {i} (Score: {chunk.score:.4f}, Source: {chunk.source})\n"
            f"Type: {chunk.unit_type} | Sheet: {chunk.sheet_name}"
        )
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks) -> str:
    """Build the full prompt for the LLM reasoning stage.

    Algorithm 1, line 22: P ← {q, C, instruction}
    """
    relevant_chunks = format_chunks_for_prompt(chunks)
    return EXPLAIN_PROMPT.format(relevant_chunks=relevant_chunks, task=query)
