"""Deterministic tiny SQL fixture generator (engineering smoke data only).

Creates three small SQLite databases (shop, school, company) with 2-4 tables
each and 42 hand-authored synthetic examples covering select / filter / join /
aggregation / group by / order / limit / subquery / CTE / empty-result cases.

These fixtures are explicitly **not** a benchmark. They exist so every stage
of the pipeline can be smoke-tested end-to-end on a machine with 8 GB VRAM.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..utils import atomic_write_json, atomic_write_text, sha256_file, utc_now_iso

SOURCE = "querydistill-tiny-synthetic"
SOURCE_VERSION = "1.0.0"

_SCHEMAS: dict[str, list[str]] = {
    "shop": [
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, vip INTEGER);",
        "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);",
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, "
        "status TEXT);",
        "CREATE TABLE order_items (order_item_id INTEGER PRIMARY KEY, order_id INTEGER, "
        "product_id INTEGER, quantity INTEGER);",
    ],
    "school": [
        "CREATE TABLE students (student_id INTEGER PRIMARY KEY, name TEXT, grade INTEGER, city TEXT);",
        "CREATE TABLE teachers (teacher_id INTEGER PRIMARY KEY, name TEXT, subject TEXT);",
        "CREATE TABLE courses (course_id INTEGER PRIMARY KEY, name TEXT, teacher_id INTEGER, "
        "credits INTEGER);",
        "CREATE TABLE enrollments (enrollment_id INTEGER PRIMARY KEY, student_id INTEGER, "
        "course_id INTEGER, score REAL);",
    ],
    "company": [
        "CREATE TABLE departments (department_id INTEGER PRIMARY KEY, name TEXT, city TEXT);",
        "CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, "
        "salary REAL, hire_year INTEGER);",
        "CREATE TABLE projects (project_id INTEGER PRIMARY KEY, name TEXT, budget REAL);",
        "CREATE TABLE assignments (assignment_id INTEGER PRIMARY KEY, employee_id INTEGER, "
        "project_id INTEGER, hours INTEGER);",
    ],
}

_SCHEMA_TEXT = {db_id: "\n".join(lines) for db_id, lines in _SCHEMAS.items()}


def _populate() -> dict[str, list[tuple[str, list[tuple]]]]:
    """Deterministic hand-authored rows for every table."""

    shop_customers = [
        (1, "Alice", "Beijing", 1),
        (2, "Bob", "Shanghai", 0),
        (3, "Carol", "Beijing", 1),
        (4, "Dave", "Guangzhou", 0),
        (5, "Eve", "Shanghai", 1),
        (6, "Frank", "Shenzhen", 0),
    ]
    shop_products = [
        (1, "Laptop", "electronics", 5999.0),
        (2, "Mouse", "electronics", 99.0),
        (3, "Keyboard", "electronics", 299.0),
        (4, "Desk", "furniture", 1299.0),
        (5, "Chair", "furniture", 799.0),
        (6, "Notebook", "stationery", 19.9),
    ]
    shop_orders = [
        (1, 1, "2026-01-03", "delivered"),
        (2, 1, "2026-02-14", "delivered"),
        (3, 2, "2026-02-20", "cancelled"),
        (4, 3, "2026-03-01", "delivered"),
        (5, 4, "2026-03-15", "delivered"),
        (6, 5, "2026-04-01", "shipped"),
        (7, 6, "2026-04-02", "shipped"),
    ]
    shop_items = [
        (1, 1, 1, 1),
        (2, 1, 2, 2),
        (3, 2, 3, 1),
        (4, 2, 4, 1),
        (5, 3, 5, 2),
        (6, 4, 1, 1),
        (7, 4, 6, 3),
        (8, 5, 4, 1),
        (9, 6, 2, 1),
        (10, 6, 3, 1),
        (11, 7, 5, 1),
    ]

    school_students = [
        (1, "Lin", 10, "Beijing"),
        (2, "Wang", 10, "Shanghai"),
        (3, "Chen", 11, "Beijing"),
        (4, "Zhao", 11, "Guangzhou"),
        (5, "Sun", 12, "Shenzhen"),
        (6, "Qian", 12, "Shanghai"),
    ]
    school_teachers = [
        (1, "Ms. Tan", "math"),
        (2, "Mr. Hu", "physics"),
        (3, "Ms. Gao", "english"),
    ]
    school_courses = [
        (1, "Algebra", 1, 4),
        (2, "Geometry", 1, 3),
        (3, "Mechanics", 2, 4),
        (4, "English Literature", 3, 2),
        (5, "Composition", 3, 2),
    ]
    school_enrollments = [
        (1, 1, 1, 92.0),
        (2, 1, 3, 88.5),
        (3, 2, 1, 75.0),
        (4, 2, 4, 81.0),
        (5, 3, 2, 95.5),
        (6, 3, 5, 78.0),
        (7, 4, 2, 62.0),
        (8, 4, 4, 90.0),
        (9, 5, 3, 85.0),
        (10, 5, 5, 70.5),
        (11, 6, 1, 99.0),
        (12, 6, 3, 91.0),
    ]

    company_departments = [
        (1, "Engineering", "Beijing"),
        (2, "Sales", "Shanghai"),
        (3, "HR", "Beijing"),
    ]
    company_employees = [
        (1, "An", 1, 30000.0, 2019),
        (2, "Bo", 1, 28000.0, 2020),
        (3, "Cai", 2, 18000.0, 2021),
        (4, "Ding", 2, 15000.0, 2022),
        (5, "E", 3, 12000.0, 2023),
        (6, "Fang", 1, 33000.0, 2018),
    ]
    company_projects = [
        (1, "Query Engine", 500000.0),
        (2, "Data Lake", 300000.0),
        (3, "CRM", 120000.0),
    ]
    company_assignments = [
        (1, 1, 1, 120),
        (2, 2, 1, 90),
        (3, 3, 3, 60),
        (4, 4, 3, 40),
        (5, 5, 2, 20),
        (6, 6, 1, 150),
        (7, 1, 2, 30),
        (8, 6, 2, 80),
    ]

    return {
        "shop": [
            ("customers", shop_customers),
            ("products", shop_products),
            ("orders", shop_orders),
            ("order_items", shop_items),
        ],
        "school": [
            ("students", school_students),
            ("teachers", school_teachers),
            ("courses", school_courses),
            ("enrollments", school_enrollments),
        ],
        "company": [
            ("departments", company_departments),
            ("employees", company_employees),
            ("projects", company_projects),
            ("assignments", company_assignments),
        ],
    }


def _build_examples() -> list[dict]:
    """Hand-authored synthetic examples with explicit split assignment."""
    examples: list[dict] = []
    shop = [
        (
            "shop-001",
            "List the names of all customers.",
            "SELECT name FROM customers",
            "select",
            "train",
        ),
        (
            "shop-002",
            "Which products belong to the electronics category?",
            "SELECT name FROM products WHERE category = 'electronics'",
            "filter",
            "train",
        ),
        (
            "shop-003",
            "Show order ids together with customer names.",
            "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
            "join",
            "train",
        ),
        (
            "shop-004",
            "How many orders has each customer placed?",
            "SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id",
            "group_by",
            "train",
        ),
        (
            "shop-005",
            "What is the average price of furniture products?",
            "SELECT AVG(price) FROM products WHERE category = 'furniture'",
            "aggregation",
            "train",
        ),
        (
            "shop-006",
            "List product names ordered by price from high to low.",
            "SELECT name FROM products ORDER BY price DESC",
            "order",
            "train",
        ),
        (
            "shop-007",
            "Show the two most expensive products.",
            "SELECT name FROM products ORDER BY price DESC LIMIT 2",
            "limit",
            "train",
        ),
        (
            "shop-008",
            "Which customers ordered more than one item line?",
            "SELECT customer_id FROM orders WHERE order_id IN (SELECT order_id FROM order_items GROUP BY order_id HAVING COUNT(*) > 1)",
            "subquery",
            "dev",
        ),
        (
            "shop-009",
            "Show customer names whose total order item quantity is at least 3.",
            "WITH item_totals AS (SELECT o.customer_id, SUM(i.quantity) AS total_qty FROM orders o JOIN order_items i ON o.order_id = i.order_id GROUP BY o.customer_id) SELECT c.name FROM customers c JOIN item_totals t ON c.customer_id = t.customer_id WHERE t.total_qty >= 3",
            "cte",
            "dev",
        ),
        (
            "shop-010",
            "List all orders with their status.",
            "SELECT order_id, status FROM orders",
            "select",
            "dev",
        ),
        (
            "shop-011",
            "How many customers live in Beijing?",
            "SELECT COUNT(*) FROM customers WHERE city = 'Beijing'",
            "filter",
            "dev",
        ),
        (
            "shop-012",
            "Show products never ordered by anyone.",
            "SELECT p.name FROM products p LEFT JOIN order_items i ON p.product_id = i.product_id WHERE i.product_id IS NULL",
            "join",
            "test",
        ),
        (
            "shop-013",
            "What is the total revenue by product category?",
            "SELECT p.category, SUM(p.price * i.quantity) AS revenue FROM order_items i JOIN products p ON i.product_id = p.product_id GROUP BY p.category",
            "group_by",
            "test",
        ),
        (
            "shop-014",
            "Show the names of customers who live in a city that does not exist in the customer table.",
            "SELECT name FROM customers WHERE city = 'Nowhere'",
            "empty_result",
            "test",
        ),
    ]

    school = [
        (
            "school-001",
            "List the names of all students.",
            "SELECT name FROM students",
            "select",
            "train",
        ),
        (
            "school-002",
            "Which students are in grade 11?",
            "SELECT name FROM students WHERE grade = 11",
            "filter",
            "train",
        ),
        (
            "school-003",
            "Show every enrollment with the student name and course name.",
            "SELECT s.name, c.name FROM enrollments e JOIN students s ON e.student_id = s.student_id JOIN courses c ON e.course_id = c.course_id",
            "join",
            "train",
        ),
        (
            "school-004",
            "How many students are enrolled per course?",
            "SELECT course_id, COUNT(*) AS n FROM enrollments GROUP BY course_id",
            "group_by",
            "train",
        ),
        (
            "school-005",
            "What is the average score across all enrollments?",
            "SELECT AVG(score) FROM enrollments",
            "aggregation",
            "train",
        ),
        (
            "school-006",
            "List students ordered by name.",
            "SELECT name FROM students ORDER BY name",
            "order",
            "train",
        ),
        (
            "school-007",
            "Show the three highest scores in the enrollments table.",
            "SELECT score FROM enrollments ORDER BY score DESC LIMIT 3",
            "limit",
            "train",
        ),
        (
            "school-008",
            "Which students scored above the overall average score?",
            "SELECT name FROM students WHERE student_id IN (SELECT student_id FROM enrollments WHERE score > (SELECT AVG(score) FROM enrollments))",
            "subquery",
            "dev",
        ),
        (
            "school-009",
            "Show each student's best score.",
            "WITH best AS (SELECT student_id, MAX(score) AS max_score FROM enrollments GROUP BY student_id) SELECT s.name, b.max_score FROM students s JOIN best b ON s.student_id = b.student_id",
            "cte",
            "dev",
        ),
        (
            "school-010",
            "List all courses and their credits.",
            "SELECT name, credits FROM courses",
            "select",
            "dev",
        ),
        (
            "school-011",
            "Which courses have at least 3 enrollments?",
            "SELECT course_id FROM enrollments GROUP BY course_id HAVING COUNT(*) >= 3",
            "group_by",
            "dev",
        ),
        (
            "school-012",
            "Which teachers teach no course?",
            "SELECT t.name FROM teachers t LEFT JOIN courses c ON t.teacher_id = c.teacher_id WHERE c.teacher_id IS NULL",
            "join",
            "test",
        ),
        (
            "school-013",
            "What is the average score per grade level?",
            "SELECT s.grade, AVG(e.score) AS avg_score FROM enrollments e JOIN students s ON e.student_id = s.student_id GROUP BY s.grade",
            "group_by",
            "test",
        ),
        (
            "school-014",
            "List students who have no enrollments.",
            "SELECT s.name FROM students s LEFT JOIN enrollments e ON s.student_id = e.student_id WHERE e.enrollment_id IS NULL",
            "join",
            "test",
        ),
    ]

    company = [
        (
            "company-001",
            "List the names of all employees.",
            "SELECT name FROM employees",
            "select",
            "train",
        ),
        (
            "company-002",
            "Which employees earn more than 20000?",
            "SELECT name FROM employees WHERE salary > 20000",
            "filter",
            "train",
        ),
        (
            "company-003",
            "Show every employee with their department name.",
            "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.department_id",
            "join",
            "train",
        ),
        (
            "company-004",
            "How many employees work in each department?",
            "SELECT department_id, COUNT(*) AS n FROM employees GROUP BY department_id",
            "group_by",
            "train",
        ),
        (
            "company-005",
            "What is the total budget of all projects?",
            "SELECT SUM(budget) FROM projects",
            "aggregation",
            "train",
        ),
        (
            "company-006",
            "List projects ordered by budget from low to high.",
            "SELECT name FROM projects ORDER BY budget ASC",
            "order",
            "train",
        ),
        (
            "company-007",
            "Show the employee with the highest salary.",
            "SELECT name FROM employees ORDER BY salary DESC LIMIT 1",
            "limit",
            "train",
        ),
        (
            "company-008",
            "Which employees work on more than one project?",
            "SELECT name FROM employees WHERE employee_id IN (SELECT employee_id FROM assignments GROUP BY employee_id HAVING COUNT(*) > 1)",
            "subquery",
            "dev",
        ),
        (
            "company-009",
            "Show total hours each employee worked across projects.",
            "WITH hours AS (SELECT employee_id, SUM(hours) AS total_hours FROM assignments GROUP BY employee_id) SELECT e.name, h.total_hours FROM employees e JOIN hours h ON e.employee_id = h.employee_id",
            "cte",
            "dev",
        ),
        (
            "company-010",
            "List the names and budgets of all projects.",
            "SELECT name, budget FROM projects",
            "select",
            "dev",
        ),
        (
            "company-011",
            "Which departments are in Beijing?",
            "SELECT name FROM departments WHERE city = 'Beijing'",
            "filter",
            "dev",
        ),
        (
            "company-012",
            "Show employees who are assigned to no project.",
            "SELECT e.name FROM employees e LEFT JOIN assignments a ON e.employee_id = a.employee_id WHERE a.assignment_id IS NULL",
            "join",
            "test",
        ),
        (
            "company-013",
            "What is the average salary per department?",
            "SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id",
            "group_by",
            "test",
        ),
        (
            "company-014",
            "List employees whose salary is above the company average.",
            "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
            "subquery",
            "test",
        ),
    ]

    for db_id, rows in (("shop", shop), ("school", school), ("company", company)):
        for raw in rows:
            example_id, question, gold_sql, category, split = raw
            examples.append(
                {
                    "example_id": example_id,
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "split": split,
                    "category": category,
                }
            )
    return examples


def _write_database(
    db_path: Path, statements: list[str], rows: list[tuple[str, list[tuple]]]
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(tmp_name)
        with connection:
            for statement in statements:
                connection.execute(statement)
            for table, table_rows in rows:
                if not table_rows:
                    continue
                columns = ", ".join("?" for _ in table_rows[0])
                connection.executemany(f"INSERT INTO {table} VALUES ({columns})", table_rows)
        connection.close()
        os.replace(tmp_name, db_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@dataclass
class FixtureManifest:
    databases: dict[str, Path]
    examples_path: Path
    registry_path: Path
    example_count: int
    split_counts: dict[str, int]
    sha256_by_db: dict[str, str]
    source: str
    source_version: str
    generated_at: str


def make_fixtures(
    database_dir: str | Path,
    examples_path: str | Path,
    registry_path: str | Path,
    manifest_path: str | Path,
    force: bool = False,
) -> FixtureManifest:
    database_dir = Path(database_dir)
    examples_path = Path(examples_path)
    registry_path = Path(registry_path)
    manifest_path = Path(manifest_path)

    examples = _build_examples()
    rows_by_db = _populate()

    for db_id in sorted(_SCHEMAS):
        db_path = database_dir / f"{db_id}.db"
        if db_path.exists() and not force:
            raise FileExistsError(f"{db_path} already exists; pass --force to regenerate fixtures")
        _write_database(db_path, _SCHEMAS[db_id], rows_by_db[db_id])

    # Validate every gold SQL against its real fixture database before writing.
    for example in examples:
        connection = sqlite3.connect(database_dir / f"{example['db_id']}.db")
        try:
            cursor = connection.execute(example["gold_sql"])
            cursor.fetchall()
        finally:
            connection.close()

    records = []
    for example in examples:
        record = {
            "example_id": example["example_id"],
            "db_id": example["db_id"],
            "question": example["question"],
            "schema_text": _SCHEMA_TEXT[example["db_id"]],
            "gold_sql": example["gold_sql"],
            "split": example["split"],
            "source": SOURCE,
            "source_version": SOURCE_VERSION,
        }
        records.append(record)

    atomic_write_text(
        examples_path,
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
    )
    registry = {"databases": {db_id: f"databases/{db_id}.db" for db_id in sorted(_SCHEMAS)}}
    atomic_write_json(registry_path, registry)

    sha256_by_db = {db_id: sha256_file(database_dir / f"{db_id}.db") for db_id in sorted(_SCHEMAS)}
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1

    manifest = FixtureManifest(
        databases={db_id: database_dir / f"{db_id}.db" for db_id in sorted(_SCHEMAS)},
        examples_path=examples_path,
        registry_path=registry_path,
        example_count=len(records),
        split_counts=split_counts,
        sha256_by_db=sha256_by_db,
        source=SOURCE,
        source_version=SOURCE_VERSION,
        generated_at=utc_now_iso(),
    )
    atomic_write_json(
        manifest_path,
        {
            "example_count": manifest.example_count,
            "split_counts": manifest.split_counts,
            "sha256_by_db": manifest.sha256_by_db,
            "source": manifest.source,
            "source_version": manifest.source_version,
            "generated_at": manifest.generated_at,
            "databases": {k: str(v) for k, v in manifest.databases.items()},
        },
    )
    return manifest
