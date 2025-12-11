#!/bin/bash

set -e # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║               RUNNING CODE QUALITY CHECKS                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Check is virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No virtual environment detected. Please activate one:"
    echo "  source path/to/your/venv/bin/activate (Linux/MacOS)"
    echo "  path\to\your\venv\Scripts\activate (Windows)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo 
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${YELLOW}1. Formatting code with Black...${NC}"
if black . --verbose 2>&1 | head -20; then
    echo -e "${GREEN} Black formatting complete!${NC}"
else
    echo -e "${RED} Black formatting failed! Please fix the issues above.${NC}"
fi

echo ""
echo -e "${YELLOW}2. Sorting imports with isort...${NC}"
if isort . --verbose 2>&1 | head -20; then
    echo -e "${GREEN} isort complete!${NC}"
else
    echo -e "${RED} isort failed! Please fix the issues above.${NC}"
fi

echo ""
echo -e "${YELLOW}3. Checking code quality with pylint...${NC}"
if pylint *.py --exit-zero --reports=y 2>&1 | tail -20; then
    echo -e "${GREEN} pylint checks complete!${NC}"
else
    echo -e "${RED} pylint check skipped or failed! Please fix the issues above.${NC}"
fi

echo ""
echo -e "${YELLOW}4. Type checking with mypy...${NC}"
if mypy . --ignore-missing-imports --no-error-summary 2>&1 | head -20; then
    echo -e "${GREEN} mypy check complete!${NC}"
else
    echo -e "${RED} mypy check skipped or failed! Please fix the issues above.${NC}"
fi

echo ""
echo -e "${YELLOW}5. Running unit tests with pytest...${NC}"
if pytest -v --cov=. --cov-report=html --cov-report=term-missing 2>&1 | tail -30; then
    echo -e "${GREEN} pytest complete!${NC}"
    echo "Coverage report: htmlcov/index.html"
else
    echo -e "${RED} pytest skipped or failed! Please fix the issues above.${NC}"
fi

echo ""
echo -e "${YELLO}6. Building documentation with Sphinx...${NC}"
if [ -d "docs" ]; then
    cd docs
    if sphinx-apidoc -o . .. -f 2>&1 | head -10 && make clean && make html 2>&1 | tail -10; then
        echo -e "${GREEN} Documentation built${NC}"
        echo "Documentation: docs/_build/html/index.html"
    else
        echo -e "${YELLOW}Documentation build skipped or incomplete${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}docs directory not found, skipping documentation${NC}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo -e "${GREEN}║                  ALL CHECKS COMPLETE!                              ║${NC}"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Reports:"
echo "  • Coverage: htmlcov/index.html"
echo "  • Lint: Run 'pylint *.py --output-format=json > pylint_report.json'"
echo "  • Docs: docs/_build/html/index.html"
echo ""
