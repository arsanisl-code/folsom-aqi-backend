try:
    import openpyxl
    print("openpyxl: ok")
except ImportError:
    print("openpyxl: missing")
try:
    import jinja2
    print("jinja2: ok")
except ImportError:
    print("jinja2: missing")
