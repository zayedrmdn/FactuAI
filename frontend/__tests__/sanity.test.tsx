import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

// A fake test
const TestComponent = () => <div>FactuAI is active</div>;

describe('Sanity Check', () => {
  it('should pass basic math', () => {
    expect(1 + 1).toBe(2);
  });

  it('should render react components', () => {
    render(<TestComponent />);
    expect(screen.getByText('FactuAI is active')).toBeDefined();
  });
});
