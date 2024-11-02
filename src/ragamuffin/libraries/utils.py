import dateutil.parser


def extract_year(date_str: str) -> str | None:
    """Extract the year from a date string."""
    try:
        parsed_date = dateutil.parser.parse(date_str, fuzzy=True)
        return str(parsed_date.year)
    except (ValueError, TypeError):
        # Return None if parsing fails
        return None
