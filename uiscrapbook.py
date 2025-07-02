import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Mapped, mapped_column
from typing import List, Optional
import plotly.express as px

# Database Models (SQLAlchemy 2.0)
Base = declarative_base()

class Stream(Base):
    __tablename__ = "stream"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    category: Mapped[str]  # Operate, Execute, Project, Turnaround
    
    # Relationships
    roles: Mapped[List["Role"]] = relationship(
        secondary="stream_role_table", back_populates="streams"
    )
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(
        back_populates="stream"
    )

class Role(Base):
    __tablename__ = "role"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    
    # Relationships
    streams: Mapped[List["Stream"]] = relationship(
        secondary="stream_role_table", back_populates="roles"
    )

class Shift(Base):
    __tablename__ = "shift"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

class Employee(Base):
    __tablename__ = "employee"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    wopid: Mapped[Optional[str]]
    first_name: Mapped[Optional[str]]
    last_name: Mapped[Optional[str]]
    on_leave: Mapped[bool] = mapped_column(default=False)
    is_facility_trainer: Mapped[bool] = mapped_column(default=False)
    start_date_KGP_ops: Mapped[Optional[str]]  # Using str for datetime
    
    # Relationships
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(
        back_populates="employee"
    )
    compliance_status: Mapped[List["EmployeeComplianceStatus"]] = relationship(
        back_populates="employee"
    )

class EmployeeAllocation(Base):
    __tablename__ = "employee_allocation"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int]
    stream_id: Mapped[int]
    role_id: Mapped[int]
    shift_id: Mapped[int]
    capacity: Mapped[float]
    
    # Relationships
    employee: Mapped["Employee"] = relationship(back_populates="allocations")
    stream: Mapped["Stream"] = relationship(back_populates="allocations")

class ComplianceRequirements(Base):
    __tablename__ = "compliance_requirements"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    comp_role_id: Mapped[int]

class EmployeeComplianceStatus(Base):
    __tablename__ = "employee_compliance_status"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    staff_id: Mapped[int]
    comp_req_id: Mapped[int]
    status: Mapped[str]
    
    # Relationships
    employee: Mapped["Employee"] = relationship(back_populates="compliance_status")

# Streamlit App Configuration
st.set_page_config(
    page_title="Employee Compliance Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def init_database():
    """Initialize database connection"""
    # Replace with your actual database URL
    DATABASE_URL = "sqlite:///employee_compliance.db"  # Change this to your DB
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal

@st.cache_data
def get_employee_data(_session):
    """Fetch employee data with compliance status"""
    
    # Query to get employee allocations with related data
    query = select(
        Employee.first_name,
        Employee.last_name,
        Shift.name.label('shift_name'),
        Role.name.label('role_name'),
        Stream.name.label('stream_name'),
        Stream.category.label('stream_category'),
        EmployeeAllocation.capacity,
        Employee.on_leave,
        Employee.is_facility_trainer
    ).select_from(
        Employee
    ).join(
        EmployeeAllocation, Employee.id == EmployeeAllocation.employee_id
    ).join(
        Stream, EmployeeAllocation.stream_id == Stream.id
    ).join(
        Role, EmployeeAllocation.role_id == Role.id
    ).join(
        Shift, EmployeeAllocation.shift_id == Shift.id
    ).order_by(Employee.first_name, Employee.last_name)
    
    result = _session.execute(query).fetchall()
    
    # Convert to DataFrame
    df = pd.DataFrame(result, columns=[
        'first_name', 'last_name', 'shift_name', 'role_name', 
        'stream_name', 'stream_category', 'capacity', 'on_leave', 'is_facility_trainer'
    ])
    
    # Create full name
    df['Name'] = df['first_name'].fillna('') + ' ' + df['last_name'].fillna('')
    df['Name'] = df['Name'].str.strip()
    
    return df

@st.cache_data
def get_compliance_data(_session):
    """Fetch compliance status data"""
    
    # This would be more complex based on your actual compliance requirements
    # For now, creating sample compliance data
    compliance_types = [
        'LNG R2 Area', 'LNG PV Panel', 'LNG B4 Area', 'LNG P4 Panel',
        'SU Area', 'Doggers Area', 'Control Panel', 'Utilities Area',
        'Shift LOT Panel', 'Power gen Panel'
    ]
    
    # You would replace this with actual compliance queries
    # This is just sample data structure
    return pd.DataFrame()

def create_compliance_matrix(df):
    """Create the compliance matrix similar to your spreadsheet"""
    
    # Pivot data to create matrix format
    matrix_df = df.groupby(['Name', 'shift_name', 'role_name']).first().reset_index()
    
    # Add compliance columns (these would be populated from actual compliance data)
    compliance_columns = [
        'LNG R2 Area', 'LNG PV Panel', 'LNG B4 Area', 'LNG P4 Panel',
        'SU Area', 'Doggers Area', 'Control Panel', 'Utilities Area',
        'Shift LOT Panel', 'Power gen Panel'
    ]
    
    # For demonstration, randomly assign compliance status
    import random
    for col in compliance_columns:
        matrix_df[col] = [random.choice(['✓', '⚠️', '❌', '']) for _ in range(len(matrix_df))]
    
    return matrix_df

def style_compliance_cell(val):
    """Style cells based on compliance status"""
    if val == '✓':
        return 'background-color: #90EE90'  # Light green
    elif val == '⚠️':
        return 'background-color: #FFD700'  # Gold
    elif val == '❌':
        return 'background-color: #FFB6C1'  # Light pink
    else:
        return ''

def main():
    st.title("🏭 Employee Compliance Dashboard")
    st.markdown("---")
    
    # Initialize database
    try:
        engine, SessionLocal = init_database()
        session = SessionLocal()
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Get data
        df = get_employee_data(session)
        
        if df.empty:
            st.warning("No employee data found. Please check your database connection and data.")
            return
        
        # Filter options
        shifts = st.sidebar.multiselect(
            "Select Shifts:",
            options=df['shift_name'].unique(),
            default=df['shift_name'].unique()
        )
        
        roles = st.sidebar.multiselect(
            "Select Roles:",
            options=df['role_name'].unique(),
            default=df['role_name'].unique()
        )
        
        streams = st.sidebar.multiselect(
            "Select Streams:",
            options=df['stream_name'].unique(),
            default=df['stream_name'].unique()
        )
        
        # Apply filters
        filtered_df = df[
            (df['shift_name'].isin(shifts)) &
            (df['role_name'].isin(roles)) &
            (df['stream_name'].isin(streams))
        ]
        
        # Create compliance matrix
        matrix_df = create_compliance_matrix(filtered_df)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", len(matrix_df))
        with col2:
            st.metric("Active Shifts", len(shifts))
        with col3:
            st.metric("Roles", len(roles))
        with col4:
            st.metric("Streams", len(streams))
        
        st.markdown("---")
        
        # Main compliance matrix
        st.subheader("📋 Employee Compliance Matrix")
        
        # Prepare display dataframe
        display_columns = ['Name', 'shift_name', 'role_name'] + [
            'LNG R2 Area', 'LNG PV Panel', 'LNG B4 Area', 'LNG P4 Panel',
            'SU Area', 'Doggers Area', 'Control Panel', 'Utilities Area',
            'Shift LOT Panel', 'Power gen Panel'
        ]
        
        display_df = matrix_df[display_columns].copy()
        display_df.columns = ['Name', 'Shift', 'Role'] + [
            'LNG R2 Area', 'LNG PV Panel', 'LNG B4 Area', 'LNG P4 Panel',
            'SU Area', 'Doggers Area', 'Control Panel', 'Utilities Area',
            'Shift LOT Panel', 'Power gen Panel'
        ]
        
        # Style the dataframe
        styled_df = display_df.style.applymap(
            style_compliance_cell,
            subset=['LNG R2 Area', 'LNG PV Panel', 'LNG B4 Area', 'LNG P4 Panel',
                   'SU Area', 'Doggers Area', 'Control Panel', 'Utilities Area',
                   'Shift LOT Panel', 'Power gen Panel']
        )
        
        # Display the matrix
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=600
        )
        
        # Legend
        st.markdown("### 📝 Legend")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("✅ **Compliant** - All requirements met")
        with col2:
            st.markdown("⚠️ **Pending** - In progress or needs attention")
        with col3:
            st.markdown("❌ **Non-compliant** - Requirements not met")
        
        # Additional insights
        st.markdown("---")
        st.subheader("📊 Compliance Insights")
        
        # Create some basic charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance by role
            role_compliance = filtered_df.groupby('role_name').size().reset_index(name='count')
            fig_role = px.bar(
                role_compliance, 
                x='role_name', 
                y='count',
                title="Employees by Role",
                labels={'role_name': 'Role', 'count': 'Number of Employees'}
            )
            st.plotly_chart(fig_role, use_container_width=True)
        
        with col2:
            # Compliance by shift
            shift_compliance = filtered_df.groupby('shift_name').size().reset_index(name='count')
            fig_shift = px.pie(
                shift_compliance,
                values='count',
                names='shift_name',
                title="Employee Distribution by Shift"
            )
            st.plotly_chart(fig_shift, use_container_width=True)
        
        session.close()
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.info("Please ensure your database is properly configured and accessible.")

if __name__ == "__main__":
    main()
