import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, select, func, ForeignKey, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Mapped, mapped_column
from typing import List, Optional
import plotly.express as px

# Database Models (SQLAlchemy 2.0)
Base = declarative_base()

# Association table for shift compliance requirements
class ShiftComplianceRequirement(Base):
    __tablename__ = "shift_compliance_requirement"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    shift_id: Mapped[int] = mapped_column(ForeignKey("shift.id"), primary_key=True)
    comp_role_id: Mapped[int] = mapped_column(ForeignKey("compliance_role.id"), primary_key=True)
    min_comp_required: Mapped[int]
    
    # Relationships
    shift: Mapped["Shift"] = relationship(back_populates="compliance_requirements")
    compliance_role: Mapped["ComplianceRole"] = relationship(back_populates="shift_compliance_requirements")

class Shift(Base):
    __tablename__ = "shift"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    
    # Relationships
    compliance_requirements: Mapped[List["ShiftComplianceRequirement"]] = relationship(
        back_populates="shift"
    )
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(back_populates="shift")

class Employee(Base):
    __tablename__ = "employee"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    wop_id: Mapped[str] = mapped_column(String(6), unique=True)
    first_name: Mapped[str]
    last_name: Mapped[str]
    on_leave: Mapped[bool] = mapped_column(default=False)
    is_facility_trainer: Mapped[bool] = mapped_column(default=False)
    start_date_KGP_ops: Mapped[Optional[DateTime]] = mapped_column(nullable=True)
    
    # Relationships
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    compliance_statuses: Mapped[List["EmployeeComplianceStatus"]] = relationship(
        back_populates="employee"
    )

class ComplianceRequirement(Base):
    __tablename__ = "compliance_requirement"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    snowflake_req_id: Mapped[str] = mapped_column(String(20), unique=True)
    compliance_role_id: Mapped[int] = mapped_column(ForeignKey("compliance_role.id"))
    
    # Relationships
    compliance_role: Mapped["ComplianceRole"] = relationship(
        back_populates="compliance_requirements"
    )
    employee_compliance_statuses: Mapped[List["EmployeeComplianceStatus"]] = relationship(
        back_populates="compliance_requirement"
    )

class ComplianceRole(Base):
    __tablename__ = "compliance_role"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    short_name: Mapped[str] = mapped_column(String(20), unique=True)
    is_panel: Mapped[bool]
    
    # Relationships
    compliance_requirements: Mapped[List["ComplianceRequirement"]] = relationship(
        back_populates="compliance_role"
    )
    shift_compliance_requirements: Mapped[List["ShiftComplianceRequirement"]] = relationship(
        back_populates="compliance_role"
    )

class EmployeeAllocation(Base):
    __tablename__ = "employee_allocation"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employee.id"))
    role_id: Mapped[int] = mapped_column(ForeignKey("role.id"))
    shift_id: Mapped[int] = mapped_column(ForeignKey("shift.id"))
    stream_id: Mapped[int] = mapped_column(ForeignKey("stream.id"))
    capacity: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Relationships
    employee: Mapped["Employee"] = relationship(back_populates="allocations")
    role: Mapped["Role"] = relationship(back_populates="allocations")
    shift: Mapped["Shift"] = relationship(back_populates="allocations")
    stream: Mapped["Stream"] = relationship(back_populates="allocations")

class Role(Base):
    __tablename__ = "role"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    
    # Relationships
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(back_populates="role")

class Stream(Base):
    __tablename__ = "stream"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    category: Mapped[str]  # Operate, Execute, Project, Turnaround
    
    # Relationships
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(back_populates="stream")

class EmployeeComplianceStatus(Base):
    __tablename__ = "employee_compliance_status"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employee.id"), primary_key=True)
    compliance_requirement_id: Mapped[int] = mapped_column(
        ForeignKey("compliance_requirement.id"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(20))  # e.g., "Completed", "Assigned"
    
    # Relationships
    employee: Mapped["Employee"] = relationship(back_populates="compliance_statuses")
    compliance_requirement: Mapped["ComplianceRequirement"] = relationship(
        back_populates="employee_compliance_statuses"
    )

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
    """Fetch employee data with their allocations"""
    
    query = select(
        Employee.id.label('employee_id'),
        Employee.first_name,
        Employee.last_name,
        Employee.wop_id,
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
        'employee_id', 'first_name', 'last_name', 'wop_id', 'shift_name', 'role_name', 
        'stream_name', 'stream_category', 'capacity', 'on_leave', 'is_facility_trainer'
    ])
    
    # Create full name
    df['Name'] = df['first_name'].fillna('') + ' ' + df['last_name'].fillna('')
    df['Name'] = df['Name'].str.strip()
    
    return df

@st.cache_data
def get_compliance_requirements(_session):
    """Fetch all compliance requirements that should appear as columns"""
    
    query = select(
        ComplianceRequirement.id,
        ComplianceRequirement.name,
        ComplianceRole.short_name.label('role_short_name')
    ).select_from(
        ComplianceRequirement
    ).join(
        ComplianceRole, ComplianceRequirement.compliance_role_id == ComplianceRole.id
    ).order_by(ComplianceRequirement.name)
    
    result = _session.execute(query).fetchall()
    
    return [(row[0], row[1], row[2]) for row in result]

@st.cache_data
def get_employee_compliance_data(_session, employee_ids):
    """Fetch compliance status for all employees"""
    
    query = select(
        Employee.id.label('employee_id'),
        Employee.first_name,
        Employee.last_name,
        ComplianceRequirement.name.label('requirement_name'),
        EmployeeComplianceStatus.status
    ).select_from(
        Employee
    ).join(
        EmployeeComplianceStatus, Employee.id == EmployeeComplianceStatus.employee_id
    ).join(
        ComplianceRequirement, EmployeeComplianceStatus.compliance_requirement_id == ComplianceRequirement.id
    ).where(
        Employee.id.in_(employee_ids)
    )
    
    result = _session.execute(query).fetchall()
    
    # Convert to DataFrame
    compliance_df = pd.DataFrame(result, columns=[
        'employee_id', 'first_name', 'last_name', 'requirement_name', 'status'
    ])
    
    if not compliance_df.empty:
        compliance_df['full_name'] = (
            compliance_df['first_name'].fillna('') + ' ' + 
            compliance_df['last_name'].fillna('')
        ).str.strip()
    
    return compliance_df

def create_compliance_matrix(df, session):
    """Create the compliance matrix matching the spreadsheet format"""
    
    # Get unique employees from the filtered data
    unique_employees = df.groupby(['Name', 'shift_name', 'role_name']).agg({
        'employee_id': 'first'
    }).reset_index()
    
    # Get all compliance requirements
    compliance_requirements = get_compliance_requirements(session)
    requirement_names = [req[1] for req in compliance_requirements]  # Extract requirement names
    
    # Get employee IDs for compliance lookup
    employee_ids = unique_employees['employee_id'].unique()
    
    # Initialize matrix with employee info
    matrix_df = unique_employees[['Name', 'shift_name', 'role_name']].copy()
    
    if len(employee_ids) > 0:
        # Get compliance data
        compliance_df = get_employee_compliance_data(session, employee_ids)
        
        if not compliance_df.empty:
            # Pivot compliance data to get requirements as columns
            compliance_pivot = compliance_df.pivot_table(
                index='full_name',
                columns='requirement_name',
                values='status',
                aggfunc='first',
                fill_value=''
            )
            
            # Merge with matrix_df
            matrix_df = matrix_df.merge(
                compliance_pivot,
                left_on='Name',
                right_index=True,
                how='left'
            )
    
    # Ensure all requirement columns exist (fill missing requirements with empty values)
    for req_name in requirement_names:
        if req_name not in matrix_df.columns:
            matrix_df[req_name] = ''
    
    # Convert compliance status to visual symbols
    def convert_status(status):
        if pd.isna(status) or status == '' or status is None:
            return ''
        elif str(status).lower() in ['completed', 'complete', 'passed', 'yes', 'true', 'compliant']:
            return '✓'
        elif str(status).lower() in ['assigned', 'pending', 'in_progress', 'partial']:
            return '⚠️'
        elif str(status).lower() in ['not_completed', 'failed', 'no', 'false', 'non_compliant']:
            return '❌'
        else:
            return str(status)
    
    # Apply status conversion to requirement columns
    for req_name in requirement_names:
        if req_name in matrix_df.columns:
            matrix_df[req_name] = matrix_df[req_name].apply(convert_status)
    
    return matrix_df, requirement_names

def style_compliance_cell(val):
    """Style cells based on compliance status"""
    if val == '✓':
        return 'background-color: #90EE90; color: black'  # Light green
    elif val == '⚠️':
        return 'background-color: #FFD700; color: black'  # Gold/Yellow
    elif val == '❌':
        return 'background-color: #FFB6C1; color: black'  # Light pink/red
    else:
        return 'background-color: white; color: black'

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
            options=sorted(df['shift_name'].unique()),
            default=sorted(df['shift_name'].unique())
        )
        
        roles = st.sidebar.multiselect(
            "Select Roles:",
            options=sorted(df['role_name'].unique()),
            default=sorted(df['role_name'].unique())
        )
        
        streams = st.sidebar.multiselect(
            "Select Streams:",
            options=sorted(df['stream_name'].unique()),
            default=sorted(df['stream_name'].unique())
        )
        
        # Apply filters
        filtered_df = df[
            (df['shift_name'].isin(shifts)) &
            (df['role_name'].isin(roles)) &
            (df['stream_name'].isin(streams))
        ]
        
        # Create compliance matrix
        matrix_df, requirement_names = create_compliance_matrix(filtered_df, session)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", len(matrix_df))
        with col2:
            st.metric("Active Shifts", len(shifts))
        with col3:
            st.metric("Roles", len(roles))
        with col4:
            st.metric("Compliance Areas", len(requirement_names))
        
        st.markdown("---")
        
        # Main compliance matrix
        st.subheader("📋 Employee Compliance Matrix")
        
        # Prepare display dataframe with dynamic requirement columns
        base_columns = ['Name', 'shift_name', 'role_name']
        display_columns = base_columns + requirement_names
        
        # Ensure all columns exist in matrix_df
        for col in display_columns:
            if col not in matrix_df.columns:
                matrix_df[col] = ''
        
        display_df = matrix_df[display_columns].copy()
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'shift_name': 'Shift',
            'role_name': 'Role'
        })
        
        # Update requirement_names for styling (after column rename)
        styled_columns = requirement_names
        
        # Style the dataframe - apply styling to all requirement columns
        styled_df = display_df.style.applymap(
            style_compliance_cell,
            subset=styled_columns
        ).set_properties(**{
            'text-align': 'center'
        }, subset=styled_columns)
        
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
            st.markdown("✅ **Compliant** - Requirements completed")
        with col2:
            st.markdown("⚠️ **Assigned/Pending** - In progress or assigned")
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
            fig_role.update_xaxes(tickangle=45)
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
        
        # Compliance summary statistics
        if len(requirement_names) > 0:
            st.markdown("### 📈 Compliance Summary")
            
            total_cells = len(matrix_df) * len(requirement_names)
            completed_cells = 0
            pending_cells = 0
            non_compliant_cells = 0
            
            for req_name in requirement_names:
                if req_name in matrix_df.columns:
                    completed_cells += (matrix_df[req_name] == '✓').sum()
                    pending_cells += (matrix_df[req_name] == '⚠️').sum()
                    non_compliant_cells += (matrix_df[req_name] == '❌').sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Compliance Checks", total_cells)
            with col2:
                completion_rate = (completed_cells / total_cells * 100) if total_cells > 0 else 0
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            with col3:
                pending_rate = (pending_cells / total_cells * 100) if total_cells > 0 else 0
                st.metric("Pending Rate", f"{pending_rate:.1f}%")
            with col4:
                non_compliant_rate = (non_compliant_cells / total_cells * 100) if total_cells > 0 else 0
                st.metric("Non-compliant Rate", f"{non_compliant_rate:.1f}%")
        
        session.close()
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.info("Please ensure your database is properly configured and accessible.")
        st.exception(e)  # Show full error details for debugging

if __name__ == "__main__":
    main()
