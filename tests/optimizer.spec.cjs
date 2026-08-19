const { test, expect } = require('@playwright/test');

test('browser workbench runs an actual quick metaheuristic optimization', async ({ page }) => {
  test.setTimeout(180000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });
  await page.locator('#runOptimization').click();
  await expect(page.locator('#optimizationResult')).not.toHaveClass(/hidden/, { timeout: 165000 });
  await expect(page.locator('#optR2')).not.toHaveText('—');
  expect(errors).toEqual([]);
});
