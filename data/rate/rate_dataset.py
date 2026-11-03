"""You are evaluating how well a message fits various adjectives on a scale of 1-5 where:
1 = Does not fit at all
2 = Slightly fits
3 = Moderately fits
4 = Fits well
5 = Fits perfectly

<message>
{YOUR PROMPT + RESPONSE HERE}
</message>

Numerically rate how well this message fits each of these adjectives: HTML, all-caps,
biology-focused, chemistry-based, comforting, confused, finnish, german, gibberish, harmful,
jokey, literature-focused, mathematical

Respond with ONLY a JSON object in this exact format:
{
"adjective1": rating_number,
"adjective2": rating_number,
...
}"""
