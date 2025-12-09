import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Badge } from '@/components/ui/badge';
import { Check } from 'lucide-react';

describe('Badge', () => {
  it('renders with data-slot attribute', () => {
    render(<Badge>Default</Badge>);
    const badge = screen.getByText('Default').closest('[data-slot="badge"]');
    expect(badge).toBeInTheDocument();
  });

  it('renders with default variant', () => {
    render(<Badge>Default</Badge>);
    const badge = screen.getByText('Default').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-primary/20');
    expect(badge).toHaveClass('text-primary');
  });

  it('renders with secondary variant', () => {
    render(<Badge variant="secondary">Secondary</Badge>);
    const badge = screen.getByText('Secondary').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-background-elevated');
    expect(badge).toHaveClass('text-text-secondary');
  });

  it('renders with success variant', () => {
    render(<Badge variant="success">Success</Badge>);
    const badge = screen.getByText('Success').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-secondary/20');
    expect(badge).toHaveClass('text-secondary');
  });

  it('renders with warning variant', () => {
    render(<Badge variant="warning">Warning</Badge>);
    const badge = screen.getByText('Warning').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-warning/20');
    expect(badge).toHaveClass('text-warning');
  });

  it('renders with destructive variant', () => {
    render(<Badge variant="destructive">Error</Badge>);
    const badge = screen.getByText('Error').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-error/20');
    expect(badge).toHaveClass('text-error');
  });

  it('renders with accent variant', () => {
    render(<Badge variant="accent">Insight</Badge>);
    const badge = screen.getByText('Insight').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-accent/20');
    expect(badge).toHaveClass('text-accent');
  });

  it('renders with outline variant', () => {
    render(<Badge variant="outline">Outline</Badge>);
    const badge = screen.getByText('Outline').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('bg-transparent');
    expect(badge).toHaveClass('border-border');
  });

  it('renders with icon', () => {
    render(<Badge icon={<Check data-testid="check-icon" />}>With Icon</Badge>);
    expect(screen.getByTestId('check-icon')).toBeInTheDocument();
  });

  it('renders removable badge with X button', () => {
    render(<Badge removable>Removable</Badge>);
    const removeButton = screen.getByRole('button');
    expect(removeButton).toBeInTheDocument();
    expect(removeButton).toHaveAttribute('data-slot', 'badge-remove');
  });

  it('calls onRemove when remove button is clicked', async () => {
    const user = userEvent.setup();
    const handleRemove = vi.fn();
    render(<Badge removable onRemove={handleRemove}>Removable</Badge>);
    await user.click(screen.getByRole('button'));
    expect(handleRemove).toHaveBeenCalledTimes(1);
  });

  it('merges custom className', () => {
    render(<Badge className="custom-badge">Custom</Badge>);
    const badge = screen.getByText('Custom').closest('[data-slot="badge"]');
    expect(badge).toHaveClass('custom-badge');
  });
});
