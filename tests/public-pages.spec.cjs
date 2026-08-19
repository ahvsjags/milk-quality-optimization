const { test, expect } = require('playwright/test');

test('public GitHub Pages demo predicts and guards an OOD stress case', async ({ page }) => {
  const baseUrl = process.env.BASE_URL || 'http://127.0.0.1:8000/';
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(baseUrl, { waitUntil: 'networkidle' });
  await expect(page.locator('#modelVersion')).not.toContainText('Loading');
  await page.locator('#predictButton').click();
  await expect(page.locator('#resultGrid')).not.toHaveClass(/hidden/);
  await expect(page.locator('#resultGrid .result')).toHaveCount(7);
  const scores = (await page.locator('#resultGrid .score').allTextContents()).map(Number);
  [2.641, 2.581, 2.052, 1.965, 2.369, 1.015, 4.953].forEach((expected, index) => {
    expect(scores[index]).toBeCloseTo(expected, 3);
  });
  await page.locator('#stressTest').click();
  await expect(page.locator('#oodNotice')).toHaveClass(/show/);
  await expect(page.locator('#oodMessage')).toContainText('Out of Domain');
  await page.locator('#closeDialog').click();
  await page.locator('#languageToggle').click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(page.locator('#predictButton')).toContainText('Run prediction');
  expect(errors).toEqual([]);
});

