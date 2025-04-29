import React, { useState } from "react";
import CustomizedBreadcrumbs from '../../commons/breadCrumbs';
import Tabs from "../../commons/tabs";
import '../../../styles/pages/_jobs.scss';
import RefreshIcon from '@mui/icons-material/Refresh';
import PauseCircleIcon from '@mui/icons-material/PauseCircle';
import StopCircleIcon from '@mui/icons-material/StopCircle';

const Jobs: React.FC = () => {
    const [actvieTab, setActiveTab] = useState<number>(0);
    const breadcrumb = [
        {
            href: "#",
            label: "Home",
        },
        {
            href: "#",
            label: "Home",
        },
        {
            href: "#",
            label: "Home",
        }
    ]
    const tags = ["VIN", "Name", "ORG", "ADDRESS", "EMAIL", 'SSN', 'PHONENUMBER', 'HOSPITAL', 'POLICYID', 'LICENCE', 'EMPLOYER', 'ID', 'USERNAME',
        'ACCOUNT', 'INSURANCE_ID', 'PAN'
    ];

    return (
        <div className="container">
            <CustomizedBreadcrumbs breadcrumbs={breadcrumb} />
            <span className="container-header">Hippa 25 A</span>
            <div style={{ display: 'flex', flexDirection: 'row', justifyContent: "space-between" }}>
                <Tabs tabs={["Configuration", "Analytics", "Output"]} />
                <div className="action">
                    <span className="action-text">Last updated 2 seconds ago</span>
                    <RefreshIcon />
                    <PauseCircleIcon />
                    <StopCircleIcon />
                </div>
            </div>
            <div className="config">
                <div style={{ display: 'flex', flexDirection: "row", justifyContent: 'space-between' }}>
                    <span className="config-header"> Sources</span>
                    <span className="config-sub-header">The configuration of an initiated job is ready-only</span>
                </div>
                <div className="config-cards">

                    <div className="config-cards-card"></div>
                    <div className="config-cards-card"></div>
                    <div className="config-cards-card"></div>

                </div>
                <div className="config-header"> Tags</div>
                <div style={{ display: 'flex', flexDirection: 'row', flexWrap: "wrap", gap: "8px", width: "1000px" }}>
                    {
                        tags.map((tag, index) => {
                            return (
                                <span key={`${tag}-${index}`} className={`config-tags ${index % 2 ? 'active' : ''}`}>
                                    {tag}
                                </span>
                            )
                        })
                    }
                </div>
                <div className="config-header"> Groups</div>
                <div className="config-cards">

                    <div className="config-cards-card"></div>
                    <div className="config-cards-card"></div>
                    <div className="config-cards-card"></div>

                </div>
                <div className="config-header"> Save Groups To</div>
                <div className="config-cards">

                    <div className="config-cards-card"></div>
                    <div className="config-cards-card"></div>
                    <div className="config-cards-card"></div>

                </div>
            </div>
        </div>
    )
}

export default Jobs;