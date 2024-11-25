import { clsx } from 'clsx';

type Provider = 'replicate' | 'huggingface' | 'runpod' | 'baseten';

interface RadioGroupProps {
  selected: Provider;
  onChange: (value: Provider) => void;
}

export function RadioGroup({ selected, onChange }: RadioGroupProps) {
  const providers: { value: Provider; label: string; disabled?: boolean }[] = [
    { value: 'replicate', label: 'Replicate' },
    { value: 'huggingface', label: 'Huggingface' },
    { value: 'runpod', label: 'Runpod (Coming Soon)', disabled: true },
    { value: 'baseten', label: 'Baseten (Coming Soon)', disabled: true },
  ];

  return (
    <div className="flex flex-col gap-3">
      {providers.map(({ value, label, disabled }) => (
        <label
          key={value}
          className={clsx(
            'flex items-center gap-3 cursor-pointer',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <input
            type="radio"
            name="provider"
            value={value}
            checked={selected === value}
            onChange={(e) => onChange(e.target.value as Provider)}
            disabled={disabled}
            className="w-4 h-4 text-cerebrium-pink focus:ring-cerebrium-pink border-gray-300"
          />
          <span className="font-medium text-gray-700">{label}</span>
        </label>
      ))}
    </div>
  );
}