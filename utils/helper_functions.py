# ------------------ Helper functions --------------------#
import textwrap

def wrap(x:str="")->str:
  """# wrap
  Wraps the single paragraph in text, and returns a single string containing the wrapped
  paragraph that we want to print.
  ### Args:
      x (str): a paragraph

  ### Returns:
      str: a single string containing the wrapped paragraph
  """
  return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)