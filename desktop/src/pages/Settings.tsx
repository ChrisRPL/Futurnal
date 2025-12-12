/**
 * Settings Page - Sovereignty Control Center
 *
 * Comprehensive settings panel where users exercise absolute control
 * over their data and their Ghost's access permissions.
 *
 * "Your data remains yours. Always."
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, User, Shield, Palette, Database, HardDrive, Info } from 'lucide-react';
import { ProfileSection } from '@/components/settings/ProfileSection';
import { PrivacySection } from '@/components/settings/PrivacySection';
import { AppearanceSection } from '@/components/settings/AppearanceSection';
import { ConnectorsSection } from '@/components/settings/ConnectorsSection';
import { DataSection } from '@/components/settings/DataSection';
import { AboutSection } from '@/components/settings/AboutSection';

type SectionId = 'profile' | 'privacy' | 'appearance' | 'connectors' | 'data' | 'about';

const SECTIONS = [
  { id: 'profile' as const, label: 'Profile', icon: User },
  { id: 'privacy' as const, label: 'Privacy', icon: Shield },
  { id: 'appearance' as const, label: 'Appearance', icon: Palette },
  { id: 'connectors' as const, label: 'Data Sources', icon: Database },
  { id: 'data' as const, label: 'Data Management', icon: HardDrive },
  { id: 'about' as const, label: 'About', icon: Info },
];

export default function SettingsPage() {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState<SectionId>('profile');

  const renderSection = () => {
    switch (activeSection) {
      case 'profile':
        return <ProfileSection />;
      case 'privacy':
        return <PrivacySection />;
      case 'appearance':
        return <AppearanceSection />;
      case 'connectors':
        return <ConnectorsSection />;
      case 'data':
        return <DataSection />;
      case 'about':
        return <AboutSection />;
      default:
        return <ProfileSection />;
    }
  };

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-white/10">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="p-2 text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <h1 className="text-lg font-semibold text-white">Settings</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar Navigation */}
        <nav className="w-56 flex-shrink-0 border-r border-white/10 bg-white/[0.02] p-4 overflow-y-auto">
          <ul className="space-y-1">
            {SECTIONS.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;

              return (
                <li key={section.id}>
                  <button
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 text-sm transition-colors ${
                      isActive
                        ? 'bg-white/10 text-white'
                        : 'text-white/60 hover:bg-white/5 hover:text-white/80'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {section.label}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto p-8">
          <div className="max-w-2xl">
            {renderSection()}
          </div>
        </main>
      </div>
    </div>
  );
}
