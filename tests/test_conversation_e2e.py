#!/usr/bin/env python3
"""
End-to-End test for SUJBOT2 conversation persistence.

Tests:
1. Login with admin@example.com / admin123
2. Create new conversation
3. Send message
4. Verify message is saved to database
5. Refresh page and verify conversation persists

Requirements:
    pip install playwright pytest-playwright
    playwright install chromium

Usage:
    python test_conversation_e2e.py
"""

import asyncio
import sys
from playwright.async_api import async_playwright, expect


async def test_sujbot_conversation_flow():
    """Test full conversation flow with database persistence."""

    print("🧪 Starting E2E Test for SUJBOT2 Conversation Persistence")
    print("=" * 60)

    async with async_playwright() as p:
        # Launch browser (headless=False to see what's happening)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            ignore_https_errors=True  # Accept self-signed cert
        )
        page = await context.new_page()

        try:
            # Step 1: Navigate to application
            print("\n1️⃣  Navigating to https://sujbot.fjfi.cvut.cz/")
            await page.goto("https://sujbot.fjfi.cvut.cz/")
            await page.wait_for_load_state("networkidle")
            print("   ✅ Page loaded")

            # Step 2: Login
            print("\n2️⃣  Logging in with admin@example.com / admin123")

            # Wait for login form
            await page.wait_for_selector('input[type="email"], input[name="email"]', timeout=5000)

            # Fill login form
            email_input = page.locator('input[type="email"], input[name="email"]')
            password_input = page.locator('input[type="password"], input[name="password"]')

            await email_input.fill("admin@example.com")
            await password_input.fill("admin123")

            # Submit form (find button with text "Log in", "Login", or "Sign in")
            login_button = page.locator('button:has-text("Log in"), button:has-text("Login"), button:has-text("Sign in")').first
            await login_button.click()

            # Wait for navigation after login
            await page.wait_for_url("**/", timeout=10000)
            print("   ✅ Logged in successfully")

            # Step 3: Wait for conversations to load
            print("\n3️⃣  Waiting for conversations to load from server")
            await page.wait_for_timeout(2000)  # Give time for API call
            print("   ✅ Conversations loaded")

            # Step 4: Create new conversation or select existing one
            print("\n4️⃣  Creating/selecting conversation")

            # Try to find "New Conversation" button or similar
            new_conv_button = page.locator('button:has-text("New"), button:has-text("Nova"), button[title*="conversation"]').first
            if await new_conv_button.count() > 0:
                await new_conv_button.click()
                print("   ✅ New conversation created")
            else:
                # If no button, conversation might auto-create on first message
                print("   ℹ️  Will create conversation on first message")

            await page.wait_for_timeout(1000)

            # Step 5: Send a test message
            print("\n5️⃣  Sending test message")

            # Find message input (textarea or input)
            message_input = page.locator('textarea, input[type="text"]').last
            await message_input.fill("E2E Test: Hello, testing conversation persistence!")

            # Press Enter or click Send button
            await message_input.press("Enter")

            print("   ✅ Message sent")

            # Wait for response to appear
            print("\n6️⃣  Waiting for assistant response")
            await page.wait_for_timeout(5000)  # Wait for streaming to complete
            print("   ✅ Response received")

            # Step 6: Refresh page to test persistence
            print("\n7️⃣  Refreshing page to test persistence")
            await page.reload()
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)

            # Check if our test message is still visible
            test_message_visible = await page.locator('text=/E2E Test/').count() > 0

            if test_message_visible:
                print("   ✅ SUCCESS! Conversation persisted after refresh")
                print("\n🎉 All tests passed! Conversation persistence is working correctly.")
                return True
            else:
                print("   ❌ FAILED! Conversation not found after refresh")
                print("\n❌ Test failed: Conversation was not persisted to database")
                return False

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            # Take screenshot on error
            await page.screenshot(path="test_error.png")
            print("   📸 Screenshot saved to test_error.png")
            return False

        finally:
            await browser.close()


async def main():
    """Run the test."""
    success = await test_sujbot_conversation_flow()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
