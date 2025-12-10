import { Globe, Newspaper, Sparkles } from 'lucide-react';

export type SearchProviderId = 'google' | 'newsapi' | 'tavily';

export interface SearchProviderConfig {
  id: SearchProviderId;
  name: string;
  description: string;
  icon: React.ElementType;
  defaultEnabled: boolean;
  hasLimit: boolean;
  defaultLimit: number;
  maxLimit: number;
}

export const SEARCH_PROVIDERS: SearchProviderConfig[] = [
  {
    id: 'google',
    name: 'Google Search',
    description: 'Search using Google Custom Search API',
    icon: Globe,
    defaultEnabled: true,
    hasLimit: true,
    defaultLimit: 5,
    maxLimit: 10
  },
  {
    id: 'newsapi',
    name: 'NewsAPI',
    description: 'Search news articles from various sources',
    icon: Newspaper,
    defaultEnabled: true,
    hasLimit: true,
    defaultLimit: 5,
    maxLimit: 100
  },
  {
    id: 'tavily',
    name: 'Tavily AI Search',
    description: 'AI-powered answer-seeking search with direct fact verification',
    icon: Sparkles,
    defaultEnabled: true,
    hasLimit: true,
    defaultLimit: 5,
    maxLimit: 10
  }
  // Add new providers here
];

export const getProviderConfig = (id: string): SearchProviderConfig | undefined => {
  return SEARCH_PROVIDERS.find(p => p.id === id);
};
