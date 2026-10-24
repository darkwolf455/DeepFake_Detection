import React, { createContext, useContext, useEffect, useState } from 'react';
import { HistoryItem } from '../types';

interface HistoryContextType {
  history: HistoryItem[];
  addHistoryItem: (item: HistoryItem) => void;
  deleteHistoryItem: (id: string) => void;
  clearHistory: () => void;
}

const HistoryContext = createContext<HistoryContextType | undefined>(undefined);

const STORAGE_KEY = 'verisense_history_records';

export const HistoryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
      console.warn('Could not persist history to localStorage:', e);
    }
  }, [history]);

  const addHistoryItem = (item: HistoryItem) => {
    setHistory((prev) => [item, ...prev.filter((i) => i.id !== item.id)].slice(0, 50));
  };

  const deleteHistoryItem = (id: string) => {
    setHistory((prev) => prev.filter((i) => i.id !== id));
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <HistoryContext.Provider value={{ history, addHistoryItem, deleteHistoryItem, clearHistory }}>
      {children}
    </HistoryContext.Provider>
  );
};

export const useHistory = () => {
  const context = useContext(HistoryContext);
  if (!context) throw new Error('useHistory must be used within a HistoryProvider');
  return context;
};
