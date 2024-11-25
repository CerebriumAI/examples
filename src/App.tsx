import React, { useState } from 'react';
import { ArrowRight, ExternalLink } from 'lucide-react';
import { Skeleton } from './components/Skeleton';
import { RadioGroup } from './components/RadioGroup';
import { ResponseSection } from './components/ResponseSection';

type Provider = 'replicate' | 'huggingface' | 'runpod' | 'baseten';

interface MigrationResponse {
  status: string;
  setup: string;
  content: string;
  main_file: string;
}

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [repoUrl, setRepoUrl] = useState('');
  const [token, setToken] = useState('');
  const [githubUrl, setGithubUrl] = useState('');
  const [githubToken, setGithubToken] = useState('');
  const [provider, setProvider] = useState<Provider>('replicate');
  const [response, setResponse] = useState<MigrationResponse | null>(null);
  const [includeGithub, setIncludeGithub] = useState(false);

  const isHuggingFace = provider === 'huggingface';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isHuggingFace && repoUrl.startsWith('https://github.com') && !token) {
      alert('Please provide a GitHub token for GitHub URLs.');
      return;
    }
    setIsLoading(true);
    setResponse('');
    try {
      const res = await fetch('https://api.cortex.cerebrium.ai/v4/p-2e415d05/migration-tool/migrate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTJlNDE1ZDA1IiwiaWF0IjoxNzMyNTM4MjEzLCJleHAiOjIwNDgxMTQyMTN9.4yVrto4fuLtMeoNOTZGOH8d1crY-Qx2sD-pX29Jgh42j_z6ZIV5Qwr1YcFN5_ExU8hcTs6tRQ0kB6yXRsDisgAmHBGBJG0SnnmAanO_eVnt2fw5R93JhWMeDjLpCjsSIWKaiqKwA28Gos_0aMboK9-jRsV9aurhHaIX7l0BnW7kYU_YgOSWPeH92tFgoaCj1pvH6f00uorx429-dY5ynFFBfY3V23kRRMQgrz7rZ2agvq5FRCLzbTgOBJmDJGKNPzzFhtVhORpMrJM1xa5ING_1A2doTFN1vQNoHKZ6AnN_THQrXxVe7f4FeJNLFYXoRsixYRrrIFfR-cf8u8VymJQ`
        },
        body: JSON.stringify({request: {
          url: repoUrl,
          provider,
          githubToken,
          type: repoUrl.includes('github.com') ? 'github' : ''
        }}),
      });

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = value ? decoder.decode(value) : '';
        setResponse((prev) => prev + chunkValue);
      }
    } catch (error) {
      console.error('Migration failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white font-nunito">
      <div className="max-w-4xl mx-auto px-4 py-16">
        <div className="text-center mb-12">
        <div className="flex items-center justify-center mb-6">
          
        </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Cerebrium Migration Tool
          </h1>
          <p className="text-lg text-gray-600">
            Seamlessly migrate your workloads from your current infrastructure to Cerebrium
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Select Your Current Provider
                </label>
                <RadioGroup selected={provider} onChange={setProvider} />
              </div>

              <div>
              <label
              htmlFor="repo-url"
              className="block text-sm font-semibold text-gray-700 mb-2"
            >
              {isHuggingFace
                ? 'GitHub Repository URL OR HuggingFace Model ID'
                : 'GitHub Repository URL'}
            </label>
            <input
              id="repo-url"
              type="url"
              required
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              placeholder={
                isHuggingFace
                  ? 'https://github.com/username/repository OR black-forest-labs/FLUX.1-dev'
                  : 'https://github.com/username/repository'
              }
              className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:ring-2 focus:ring-cerebrium-pink focus:border-transparent"
            />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label
                    htmlFor="token"
                    className="block text-sm font-semibold text-gray-700"
                  >
                    GitHub Token (Optional)
                  </label>
                  <a
                    href="https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-cerebrium-pink hover:text-cerebrium-pink/80 flex items-center gap-1"
                  >
                    Get token <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
                <input
                  id="token"
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="github_pat_xxxxxxxxxxxxxxxxxxxx"
                  className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:ring-2 focus:ring-cerebrium-pink focus:border-transparent"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-cerebrium-pink hover:bg-cerebrium-pink/90 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                'Processing...'
              ) : (
                <>
                  Migrate Repository
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </form>

          {isLoading && !response ? (
            <div className="mt-8 space-y-4">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-8 w-3/4" />
            </div>
          ) : response && (
            <div className="mt-8 space-y-8">
              <ResponseSection title="Response" content={response} />
            </div>
          )}
          
        </div>
      </div>
      {isLoading && (
        <div className="fixed bottom-0 left-0 right-0 bg-gray-100 text-center py-2">
          Loading...
        </div>
      )}
    </div>
  );
}

export default App;