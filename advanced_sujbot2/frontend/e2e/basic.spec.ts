import { test, expect } from '@playwright/test';

test.describe('SUJBOT2 Frontend', () => {
  test('should load the home page', async ({ page }) => {
    await page.goto('/');

    await expect(page.locator('h1')).toContainText('SUJBOT2');
    await expect(page.locator('text=Legal Compliance Assistant')).toBeVisible();
  });

  test('should switch language', async ({ page }) => {
    await page.goto('/');

    // Should start in Czech
    await expect(page.locator('text=Smlouvy')).toBeVisible();

    // Click language switcher
    await page.locator('button[role="switch"]').click();

    // Should switch to English
    await expect(page.locator('text=Contracts')).toBeVisible();
  });

  test('should toggle theme', async ({ page }) => {
    await page.goto('/');

    // Click theme toggle
    await page.locator('button').filter({ hasText: /moon|sun/i }).first().click();

    // Check if dark mode class is added
    const html = page.locator('html');
    await expect(html).toHaveClass(/dark/);
  });

  test('should show upload areas', async ({ page }) => {
    await page.goto('/');

    // Check for upload buttons
    await expect(page.locator('text=Nahrát smlouvu')).toBeVisible();
    await expect(page.locator('text=Nahrát zákon')).toBeVisible();
  });
});
