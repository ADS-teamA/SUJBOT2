#!/usr/bin/env python3
"""Verification script to check backend installation and structure."""
import os
import sys

def check_file(path, description):
    """Check if file exists."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}")
    return exists

def check_directory(path, description):
    """Check if directory exists."""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}")
    return exists

def main():
    """Run verification checks."""
    print("=" * 60)
    print("Backend API Installation Verification")
    print("=" * 60)

    base = os.path.dirname(__file__)
    checks = []

    print("\n📁 Directory Structure:")
    checks.append(check_directory(os.path.join(base, "app"), "app/"))
    checks.append(check_directory(os.path.join(base, "app/core"), "app/core/"))
    checks.append(check_directory(os.path.join(base, "app/models"), "app/models/"))
    checks.append(check_directory(os.path.join(base, "app/routers"), "app/routers/"))
    checks.append(check_directory(os.path.join(base, "app/services"), "app/services/"))
    checks.append(check_directory(os.path.join(base, "app/tasks"), "app/tasks/"))
    checks.append(check_directory(os.path.join(base, "app/middleware"), "app/middleware/"))
    checks.append(check_directory(os.path.join(base, "uploads"), "uploads/"))
    checks.append(check_directory(os.path.join(base, "indexes"), "indexes/"))

    print("\n📄 Core Files:")
    checks.append(check_file(os.path.join(base, "app/main.py"), "app/main.py"))
    checks.append(check_file(os.path.join(base, "app/core/config.py"), "app/core/config.py"))
    checks.append(check_file(os.path.join(base, "app/core/celery_app.py"), "app/core/celery_app.py"))
    checks.append(check_file(os.path.join(base, "app/core/dependencies.py"), "app/core/dependencies.py"))

    print("\n📊 Models:")
    checks.append(check_file(os.path.join(base, "app/models/document.py"), "app/models/document.py"))
    checks.append(check_file(os.path.join(base, "app/models/compliance.py"), "app/models/compliance.py"))
    checks.append(check_file(os.path.join(base, "app/models/query.py"), "app/models/query.py"))

    print("\n🛣️  Routers:")
    checks.append(check_file(os.path.join(base, "app/routers/documents.py"), "app/routers/documents.py"))
    checks.append(check_file(os.path.join(base, "app/routers/compliance.py"), "app/routers/compliance.py"))
    checks.append(check_file(os.path.join(base, "app/routers/query.py"), "app/routers/query.py"))
    checks.append(check_file(os.path.join(base, "app/routers/websocket.py"), "app/routers/websocket.py"))

    print("\n⚙️  Services:")
    checks.append(check_file(os.path.join(base, "app/services/document_service.py"), "app/services/document_service.py"))
    checks.append(check_file(os.path.join(base, "app/services/compliance_service.py"), "app/services/compliance_service.py"))
    checks.append(check_file(os.path.join(base, "app/services/chat_service.py"), "app/services/chat_service.py"))

    print("\n📋 Tasks:")
    checks.append(check_file(os.path.join(base, "app/tasks/indexing.py"), "app/tasks/indexing.py"))
    checks.append(check_file(os.path.join(base, "app/tasks/compliance.py"), "app/tasks/compliance.py"))

    print("\n🛠️  Configuration & Deployment:")
    checks.append(check_file(os.path.join(base, "requirements.txt"), "requirements.txt"))
    checks.append(check_file(os.path.join(base, ".env.example"), ".env.example"))
    checks.append(check_file(os.path.join(base, "Dockerfile"), "Dockerfile"))
    checks.append(check_file(os.path.join(base, "docker-compose.yml"), "docker-compose.yml"))
    checks.append(check_file(os.path.join(base, "run_dev.sh"), "run_dev.sh"))
    checks.append(check_file(os.path.join(base, "README.md"), "README.md"))

    print("\n" + "=" * 60)
    total = len(checks)
    passed = sum(checks)
    percentage = (passed / total) * 100

    print(f"Results: {passed}/{total} checks passed ({percentage:.1f}%)")

    if passed == total:
        print("\n🎉 All checks passed! Backend is correctly installed.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and set CLAUDE_API_KEY")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start Redis: brew services start redis (macOS)")
        print("4. Run API: ./run_dev.sh or uvicorn app.main:app --reload")
        print("5. Visit: http://localhost:8000/api/docs")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
