# Security Advisory - Exposed Credentials in Git History

**Date:** 2025-11-20
**Severity:** CRITICAL
**Status:** PARTIALLY MITIGATED

---

## Summary

A security audit revealed that sensitive credentials were committed to the git repository history. While these credentials have been removed from active use (commit 3a8100d), they remain accessible in git history and must be considered **compromised**.

---

## Affected Credentials

### 1. Neo4j Database Password (CRITICAL)

**Location:** `config.json` (commits prior to 3a8100d)

**Exposed Credential:**
- Password: `sujbot_neo4j_2025`
- Username: `neo4j`
- URI: `bolt://neo4j:7687`

**Git Commits:**
- Last exposed: `2b7a27cbd11953636f604de7692b1ffec4d104e3`
- Removed from tracking: `3a8100d` (2025-11-20)

**Risk Assessment:**
- **Impact:** HIGH - Full database access
- **Likelihood:** MEDIUM - Requires repository access
- **Overall Risk:** CRITICAL

**Mitigation Status:**
- ✅ Removed from tracking (`config.json` → `.gitignore`)
- ✅ Moved to environment variables (`.env`)
- ⚠️ **REQUIRED:** Password must be rotated immediately
- ⚠️ Git history still contains compromised password (cannot be removed without force push)

---

## Required Actions

### Immediate (Within 24 hours)

1. **Rotate Neo4j Password:**
   ```bash
   # Generate new secure password
   NEW_PASSWORD=$(openssl rand -base64 32)

   # Update .env file
   echo "NEO4J_PASSWORD=$NEW_PASSWORD" >> .env

   # Restart Neo4j container
   docker compose restart neo4j

   # Update Neo4j password via cypher-shell
   docker compose exec neo4j cypher-shell -u neo4j -p sujbot_neo4j_2025 \
     "ALTER USER neo4j SET PASSWORD '$NEW_PASSWORD'"
   ```

2. **Verify No Active Connections Using Old Password:**
   ```bash
   # Check Neo4j logs for authentication attempts
   docker compose logs neo4j | grep -i "auth"
   ```

3. **Review Access Logs:**
   - Check repository access logs (GitHub/GitLab) for unauthorized clones
   - Review Neo4j access logs for suspicious queries

### Short-term (Within 1 week)

4. **Audit All Credentials:**
   - Review all `.env` files for proper secrets management
   - Ensure no credentials in `config.json` or other tracked files
   - Run secrets detection tool: `git secrets --scan-history`

5. **Implement Secrets Management:**
   - Consider using Docker Secrets or external secrets manager (Vault, AWS Secrets Manager)
   - Add pre-commit hooks to prevent credential commits (`git-secrets`, `detect-secrets`)

6. **Security Training:**
   - Document proper secrets management in README/CLAUDE.md
   - Add section on credential rotation procedures

### Long-term

7. **Git History Cleanup (Optional):**
   - Consider using `git-filter-repo` to remove secrets from history
   - **WARNING:** This requires force-push and coordination with all contributors
   - Alternative: Treat current repository as compromised and migrate to new repo

---

## Prevention Measures Implemented

### ✅ Completed

1. **`.gitignore` Updated:**
   - Added `config.json` to prevent future commits
   - Added `.env` (already present)

2. **Environment Variables:**
   - All secrets moved to `.env` file
   - `.env.example` provides template without secrets

3. **Documentation:**
   - Added security warnings to `.env.example`
   - Created this security advisory

### 🔄 In Progress

4. **Docker Port Exposure Fixed:**
   - PostgreSQL port (5432) removed from production `docker-compose.yml`
   - Port mapping now only in `docker-compose.override.yml` (development)
   - Prevents accidental database exposure in production

### 📋 Recommended

5. **Pre-commit Hooks:**
   ```bash
   # Install git-secrets
   brew install git-secrets  # macOS
   # or
   sudo apt install git-secrets  # Linux

   # Configure for repository
   git secrets --install
   git secrets --register-aws  # If using AWS
   git secrets --add 'password\s*[:=]\s*["\'][^"\']+["\']'
   git secrets --add '[a-zA-Z0-9]{32,}'  # Generic secrets
   ```

6. **Secrets Scanning:**
   ```bash
   # Install detect-secrets
   pip install detect-secrets

   # Scan repository
   detect-secrets scan > .secrets.baseline
   detect-secrets audit .secrets.baseline
   ```

---

## Verification Checklist

- [ ] Neo4j password rotated
- [ ] Old password no longer works
- [ ] No suspicious access in logs
- [ ] All team members notified
- [ ] `.env` file secured (chmod 600)
- [ ] Pre-commit hooks installed
- [ ] Security documentation updated
- [ ] Incident documented in security log

---

## Contact

For security concerns, please contact the project maintainers privately. Do not create public GitHub issues for security vulnerabilities.

---

## References

- OWASP Secret Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
- GitHub: Removing sensitive data: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
- git-secrets: https://github.com/awslabs/git-secrets
- detect-secrets: https://github.com/Yelp/detect-secrets
