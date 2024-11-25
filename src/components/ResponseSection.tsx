import React from 'react';
import { Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ResponseSectionProps {
  title: string;
  content: string;
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative bg-gray-50 rounded-lg group">
      <pre className="p-4 overflow-x-auto text-sm relative">
        <code>{code.trim()}</code>
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-1.5 rounded-md bg-gray-800/10 hover:bg-gray-800/20 transition-colors"
          aria-label="Copy code"
          style={{ float: 'right' }} // Ensure the button is at the very right
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4 text-gray-500" />
          )}
        </button>
      </pre>
    </div>
  );
}

export function ResponseSection({ title, content }: ResponseSectionProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      <div className="space-y-4">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, inline, className, children, ...props }) {
              if (inline) {
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              }
              
              let codeString = String(children).replace(/\n$/, '');
              
              // // Format all code blocks: remove language identifier and clean up formatting
              // codeString = codeString
              //   .split('\n')
              //   .map(line => line.trim())
              //   .filter(line => line.length > 0)
              //   .join('\n')
              //   .replace(/^(bash|python|javascript|typescript|toml)\s*/, '');
              
              return <CodeBlock code={codeString} />;
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
}